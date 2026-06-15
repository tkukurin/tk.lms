# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm[proxy]",
#     "boto3",
#     "pyyaml"
# ]
# ///
"""LiteLLM proxy: single OpenAI-compatible endpoint for bedrock/openai/anthropic/gemini/ollama.

[Agenta (docker)] --> host.docker.internal:4100/v1 --> litellm --> provider APIs

Env vars:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY  - provider keys
  AWS credentials (SSO/profile/env)                  - bedrock
  OLLAMA_BASE_URL                                    - defaults to http://localhost:8888
  AGENTA_API_KEY                                     - push command only
"""
import os
import sys
import json
import socket
import argparse
import subprocess
import urllib.request
from dataclasses import dataclass

import yaml
import boto3
import litellm
from litellm.proxy.proxy_cli import run_server


@dataclass(frozen=True)
class ProxyConfig:
    """All derived URLs flow from port. Change port, everything updates."""
    host: str = "0.0.0.0"
    port: int = 4100
    api_key: str = "sk-local-proxy-key"
    ollama_base_url: str = "http://localhost:8888"
    agenta_url: str = "http://localhost/api"
    agenta_container: str = "agenta-oss-gh-services-1"

    @property
    def base_url(self): return f"http://127.0.0.1:{self.port}/v1"

    @property
    def docker_url(self): return f"http://host.docker.internal:{self.port}/v1"


def port_owner(port: int) -> str | None:
    """Process name holding the port, or None if free."""
    r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    pid = r.stdout.strip().split("\n")[0]
    r2 = subprocess.run(["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True)
    return r2.stdout.strip() or f"pid {pid}"


def agenta_configured_url(cfg: ProxyConfig) -> str | None:
    """Returns base_url agenta has stored for litellm-proxy, or None."""
    try:
        resp = json.loads(urllib.request.urlopen(f"{cfg.agenta_url}/vault/v1/secrets/", timeout=3).read())
        for s in resp:
            if s.get("header", {}).get("name") == "litellm-proxy":
                return s["secret"]["data"]["provider"]["url"]
    except (urllib.error.URLError, OSError, KeyError):
        pass
    return None


def build_config(cfg: ProxyConfig):
    """Probes providers, returns (litellm_config, active_names, warnings)."""
    model_list = []
    active = []
    warnings = []

    for name, key_var in [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")]:
        if os.environ.get(key_var):
            model_list.append({"model_name": f"{name}/*", "litellm_params": {"model": f"{name}/*"}})
            active.append(name)

    region = boto3.Session().region_name or "eu-central-1"
    model_list.append({"model_name": "bedrock/*", "litellm_params": {"model": "bedrock/*", "aws_region_name": region}})
    active.append("bedrock")

    try:
        urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=3)
        model_list.append({"model_name": "ollama/*", "litellm_params": {"model": "ollama/*", "api_base": cfg.ollama_base_url}})
        active.append("ollama")
    except (urllib.error.URLError, OSError):
        warnings.append(f"ollama: {cfg.ollama_base_url} unreachable (tunnel running?)")

    config = {"model_list": model_list, "general_settings": {"master_key": cfg.api_key}}
    return config, active, warnings


def get_live_models(cfg: ProxyConfig):
    """Query each provider API for actually available models. Returns set of canonical names."""
    live = set()

    if key := os.environ.get("OPENAI_API_KEY"):
        req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"})
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        skip = ("ft:", "dall-e", "tts", "whisper", "davinci", "babbage", "embed", "moderation", "realtime", "audio", "chatgpt-image")
        for m in resp["data"]:
            mid = m["id"]
            if any(mid.startswith(s) or s in mid for s in skip): continue
            live.add(f"openai/{mid}")

    if key := os.environ.get("GEMINI_API_KEY"):
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        skip = ("tts", "image", "live", "robotics", "lyria", "embedding", "computer-use", "deep-research", "antigravity", "nano-banana")
        for m in resp["models"]:
            name = m["name"].removeprefix("models/")
            if "generateContent" not in m.get("supportedGenerationMethods", []): continue
            if any(s in name for s in skip): continue
            live.add(f"gemini/{name}")

    if key := os.environ.get("ANTHROPIC_API_KEY"):
        req = urllib.request.Request("https://api.anthropic.com/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
            for m in resp["data"]:
                live.add(f"anthropic/{m['id']}")
        except urllib.error.HTTPError:
            pass

    session = boto3.Session()
    region = session.region_name or "eu-central-1"
    bedrock = session.client("bedrock", region_name=region)
    resp = bedrock.list_foundation_models(byOutputModality="TEXT")
    for m in resp["modelSummaries"]:
        if m.get("modelLifecycle", {}).get("status") == "ACTIVE":
            live.add(f"bedrock/eu.{m['modelId']}")

    try:
        resp = json.loads(urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=5).read())
        for m in resp.get("models", []):
            live.add(f"ollama/{m['name']}")
    except (urllib.error.URLError, OSError):
        pass

    return live


def discover_models(cfg: ProxyConfig):
    """Cross-reference live provider models with litellm cost map. Sorted by input cost."""
    live = get_live_models(cfg)
    cost_map = {}
    for name, info in litellm.model_cost.items():
        if name == "sample_spec" or info.get("mode") != "chat": continue
        provider = info.get("litellm_provider", "")
        inp = info.get("input_cost_per_token", 0)
        out = info.get("output_cost_per_token", 0)
        if inp == 0 and out == 0: continue

        if provider == "openai" and "/" not in name:
            cost_map[f"openai/{name}"] = (inp, out)
        elif provider == "anthropic" and "/" not in name and "claude" in name:
            cost_map[f"anthropic/{name}"] = (inp, out)
        elif provider == "gemini" and name.startswith("gemini/") and name.count("/") == 1:
            cost_map[name] = (inp, out)
        elif provider == "bedrock" and name.startswith("eu."):
            cost_map[f"bedrock/{name}"] = (inp, out)

    models = []
    for name in live:
        if name in cost_map:
            inp, out = cost_map[name]
            models.append((name, inp, out))
        else:
            models.append((name, 0.0, 0.0))

    models.sort(key=lambda x: (x[1] == 0, x[1]))
    return models


def cmd_serve(cfg: ProxyConfig):
    owner = port_owner(cfg.port)
    if owner:
        sys.exit(f"error: port {cfg.port} in use by: {owner}")

    config, active, warnings = build_config(cfg)

    agenta_url = agenta_configured_url(cfg)
    if agenta_url and agenta_url != cfg.docker_url:
        warnings.append(f"agenta expects {agenta_url} but proxy will serve {cfg.docker_url} — run `push` to fix")

    for w in warnings:
        print(f"  WARN  {w}")

    with open("litellm_config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"providers:  {', '.join(active)}")
    print(f"host url:   {cfg.base_url}")
    print(f"docker url: {cfg.docker_url}")
    print(f"api key:    {cfg.api_key}\n")
    run_server(args=["--config", "litellm_config.yaml", "--host", cfg.host, "--port", str(cfg.port)], standalone_mode=False)


def cmd_test(cfg: ProxyConfig):
    print(f"host test ({cfg.base_url})")
    resp = litellm.completion(
        model="bedrock/eu.amazon.nova-micro-v1:0",
        messages=[{"role": "user", "content": "Say 'hi'"}],
        max_tokens=5, api_base=cfg.base_url, api_key=cfg.api_key, custom_llm_provider="openai",
    )
    print(f"  chat: {resp.choices[0].message.content}")


def cmd_docker(cfg: ProxyConfig):
    code = (
        "import urllib.request,json;"
        f"req=urllib.request.Request('{cfg.docker_url}/chat/completions',"
        f"data=json.dumps({{'model':'bedrock/eu.amazon.nova-micro-v1:0','messages':[{{'role':'user','content':'hi'}}],'max_tokens':5}}).encode(),"
        f"headers={{'Authorization':'Bearer {cfg.api_key}','Content-Type':'application/json'}});"
        "print(json.loads(urllib.request.urlopen(req,timeout=10).read())['choices'][0]['message']['content'])"
    )
    r = subprocess.run(["docker", "exec", cfg.agenta_container, "python", "-c", code], capture_output=True, text=True)
    print(f"docker test ({cfg.docker_url})")
    if r.returncode == 0: print(f"  chat: {r.stdout.strip()}")
    else: print(f"  FAIL: {r.stderr.strip()}")


def cmd_models(cfg: ProxyConfig, top: int = 20):
    models = discover_models(cfg)
    print(f"cheapest {top} chat models (input $/1M tok):\n")
    for name, inp, out in models[:top]:
        print(f"  ${inp*1e6:>7.2f} / ${out*1e6:>7.2f}  {name}")
    print(f"\ntotal: {len(models)} models")


def find_existing_secret(cfg: ProxyConfig, agenta_api_key: str) -> str | None:
    """Returns secret ID for 'litellm-proxy' if it exists in agenta vault."""
    req = urllib.request.Request(
        f"{cfg.agenta_url}/vault/v1/secrets/",
        headers={"Authorization": f"ApiKey {agenta_api_key}"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        for s in resp:
            if s.get("header", {}).get("name") == "litellm-proxy":
                return s["id"]
    except (urllib.error.URLError, OSError, KeyError):
        pass
    return None


def cmd_push(cfg: ProxyConfig, agenta_api_key: str):
    if not agenta_api_key:
        sys.exit("need --agenta-api-key or AGENTA_API_KEY env (create in Agenta UI: Settings > API Keys)")

    all_models = discover_models(cfg)
    model_names = [m[0] for m in all_models]

    payload = {
        "header": {"name": "litellm-proxy"},
        "secret": {
            "kind": "custom_provider",
            "data": {
                "kind": "custom",
                "provider": {"url": cfg.docker_url, "key": cfg.api_key},
                "models": [{"slug": m} for m in model_names],
            },
        },
    }

    existing_id = find_existing_secret(cfg, agenta_api_key)
    if existing_id:
        url = f"{cfg.agenta_url}/vault/v1/secrets/{existing_id}"
        method = "PUT"
    else:
        url = f"{cfg.agenta_url}/vault/v1/secrets/"
        method = "POST"

    print(f"{'updating' if existing_id else 'creating'} 'litellm-proxy' ({len(model_names)} models) at {cfg.agenta_url}")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"ApiKey {agenta_api_key}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        print(f"  OK - secret id: {result.get('id', existing_id or '?')}")
        print(f"  cheapest 5:")
        for name, inp, out in all_models[:5]:
            print(f"    ${inp*1e6:.2f}/${out*1e6:.2f}  {name}")
        print(f"  ... and {len(model_names) - 5} more")
    except urllib.error.HTTPError as e:
        print(f"  FAIL ({e.code}): {e.read().decode()}")


if __name__ == "__main__":
    cfg = ProxyConfig(
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:8888"),
    )

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="start litellm proxy (default)")
    sub.add_parser("test", help="test proxy from host")
    sub.add_parser("docker", help="test from inside agenta container")
    models_p = sub.add_parser("models", help="list discovered models sorted by cost")
    models_p.add_argument("--top", type=int, default=20, help="how many to show")
    push_p = sub.add_parser("push", help="discover models and push to agenta vault")
    push_p.add_argument("--agenta-api-key", help="or set AGENTA_API_KEY env")

    args = p.parse_args()
    cmd = args.cmd or "serve"
    {
        "serve": lambda: cmd_serve(cfg),
        "test": lambda: cmd_test(cfg),
        "docker": lambda: cmd_docker(cfg),
        "models": lambda: cmd_models(cfg, top=args.top if hasattr(args, "top") else 20),
        "push": lambda: cmd_push(cfg, agenta_api_key=getattr(args, "agenta_api_key", None) or os.environ.get("AGENTA_API_KEY", "")),
    }[cmd]()
