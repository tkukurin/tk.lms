# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm[proxy]",
#     "boto3",
#     "pyyaml"
# ]
# ///
"""LiteLLM proxy: single OpenAI-compatible endpoint for bedrock/openai/anthropic/gemini.

[Agenta (docker)] --> host.docker.internal:4000/v1 --> litellm --> provider APIs

Routing: model_list wildcards map prefixes to backends.
  model="openai/gpt-4o" -> OpenAI, model="bedrock/eu.X" -> Bedrock, etc.
  Provider API keys read from env automatically.

Env vars:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY  - provider keys
  AWS credentials (SSO/profile/env)                  - bedrock
  AGENTA_API_KEY                                     - push command only

Agenta config (created by `push`):
  base_url: http://host.docker.internal:4000/v1  (must end /v1, no trailing /)
  api_key:  sk-local-proxy-key
  kind:     openai (OpenAI-compatible custom provider)
"""
import os
import sys
import json
import argparse
import subprocess
import urllib.request

import yaml
import boto3
import litellm
from litellm.proxy.proxy_cli import run_server

API_KEY = "sk-local-proxy-key"
HOST = "0.0.0.0"
PORT = 4000
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
DOCKER_URL = f"http://host.docker.internal:{PORT}/v1"
TEST_MODEL = "bedrock/eu.amazon.nova-micro-v1:0"
AGENTA_URL = "http://localhost/api"
AGENTA_CONTAINER = "agenta-oss-gh-services-1"

PROVIDERS = ["openai", "anthropic", "gemini", "bedrock"]


def get_bedrock_region():
    session = boto3.Session()
    return session.region_name or "eu-central-1"


def get_live_models():
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

    return live


def discover_models():
    """Cross-reference live provider models with litellm cost map. Sorted by input cost."""
    live = get_live_models()
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


def build_config():
    region = get_bedrock_region()
    model_list = [
        {"model_name": "bedrock/*", "litellm_params": {"model": "bedrock/*", "aws_region_name": region}},
        {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
        {"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}},
        {"model_name": "gemini/*", "litellm_params": {"model": "gemini/*"}},
    ]
    return {"model_list": model_list, "general_settings": {"master_key": API_KEY}}, region


def cmd_serve(args):
    config, region = build_config()
    with open("litellm_config.yaml", "w") as f:
        yaml.dump(config, f)

    active = [p for p in PROVIDERS if p == "bedrock" or os.environ.get(f"{p.upper()}_API_KEY")]
    print(f"providers:  {', '.join(active)} (region={region})")
    print(f"host url:   {BASE_URL}")
    print(f"docker url: {DOCKER_URL}")
    print(f"api key:    {API_KEY}\n")
    run_server(args=["--config", "litellm_config.yaml", "--host", HOST, "--port", str(PORT)], standalone_mode=False)


def cmd_test(args):
    print(f"host test ({BASE_URL})")
    resp = litellm.completion(
        model=TEST_MODEL, messages=[{"role": "user", "content": "Say 'hi'"}],
        max_tokens=5, api_base=BASE_URL, api_key=API_KEY, custom_llm_provider="openai",
    )
    print(f"  chat: {resp.choices[0].message.content}")


def cmd_docker(args):
    code = (
        "import urllib.request,json;"
        f"req=urllib.request.Request('{DOCKER_URL}/chat/completions',"
        f"data=json.dumps({{'model':'{TEST_MODEL}','messages':[{{'role':'user','content':'hi'}}],'max_tokens':5}}).encode(),"
        f"headers={{'Authorization':'Bearer {API_KEY}','Content-Type':'application/json'}});"
        "print(json.loads(urllib.request.urlopen(req,timeout=10).read())['choices'][0]['message']['content'])"
    )
    r = subprocess.run(["docker", "exec", AGENTA_CONTAINER, "python", "-c", code], capture_output=True, text=True)
    print(f"docker test ({DOCKER_URL})")
    if r.returncode == 0: print(f"  chat: {r.stdout.strip()}")
    else: print(f"  FAIL: {r.stderr.strip()}")


def cmd_models(args):
    models = discover_models()
    n = args.top or 20
    print(f"cheapest {n} chat models (input $/1M tok):\n")
    for name, inp, out in models[:n]:
        print(f"  ${inp*1e6:>7.2f} / ${out*1e6:>7.2f}  {name}")
    print(f"\ntotal: {len(models)} models across {', '.join(PROVIDERS)}")


def cmd_push(args):
    agenta_api_key = args.agenta_api_key or os.environ.get("AGENTA_API_KEY", "")
    if not agenta_api_key:
        sys.exit("need --agenta-api-key or AGENTA_API_KEY env (create in Agenta UI: Settings > API Keys)")

    all_models = discover_models()
    model_names = [m[0] for m in all_models]
    provider_name = "litellm-proxy"

    payload = {
        "header": {"name": provider_name},
        "secret": {
            "kind": "custom_provider",
            "data": {
                "kind": "openai",
                "provider": {"url": DOCKER_URL, "key": API_KEY},
                "models": [{"slug": m} for m in model_names],
            },
        },
    }

    print(f"pushing {len(model_names)} models as '{provider_name}' to {AGENTA_URL}")
    req = urllib.request.Request(
        f"{AGENTA_URL}/vault/v1/secrets/",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"ApiKey {agenta_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        print(f"  OK - secret id: {result.get('id', '?')}")
        print(f"  cheapest 5:")
        for name, inp, out in all_models[:5]:
            print(f"    ${inp*1e6:.2f}/${out*1e6:.2f}  {name}")
        print(f"  ... and {len(model_names) - 5} more")
    except urllib.error.HTTPError as e:
        print(f"  FAIL ({e.code}): {e.read().decode()}")


if __name__ == "__main__":
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
    {"serve": cmd_serve, "test": cmd_test, "docker": cmd_docker, "models": cmd_models, "push": cmd_push}[cmd](args)
