# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm[proxy]",
#     "boto3",
#     "pyyaml"
# ]
# ///
"""litellm proxy for bedrock for openai compatible api.

    uv run litellmproxy.py          # start proxy
    uv run litellmproxy.py test     # test from host
    uv run litellmproxy.py docker   # test from inside agenta docker containers

Agenta setup (Settings > Model Hub > OpenAI-Compatible):
    Base URL: http://host.docker.internal:4000/v1
    API Key:  sk-local-bedrock-key
    Model:    bedrock/eu.amazon.nova-micro-v1:0

Gotchas:
  - Agenta runs in Docker, so it can't reach localhost. Use host.docker.internal.
  - Base URL MUST end with /v1 (no trailing slash).
  - Use eu.* inference profiles (eu-central-1).
"""
import sys
import yaml
import boto3
import litellm
import subprocess
from litellm.proxy.proxy_cli import run_server


api_key = "sk-local-bedrock-key"
host = "0.0.0.0"
port = 4000
base_url = f"http://127.0.0.1:{port}/v1"
docker_url = f"http://host.docker.internal:{port}/v1"
test_model = "bedrock/eu.amazon.nova-micro-v1:0"


def test():
    print(f"host test ({base_url})")
    resp = litellm.completion(
        model=test_model,
        messages=[{"role": "user", "content": "Say 'hi'"}],
        max_tokens=5,
        api_base=base_url,
        api_key=api_key,
        custom_llm_provider="openai",
    )
    print(f"  chat: {resp.choices[0].message.content}")


def docker():
    code = (
        "import urllib.request,json;"
        f"req=urllib.request.Request('{docker_url}/chat/completions',"
        f"data=json.dumps({{'model':'{test_model}','messages':[{{'role':'user','content':'hi'}}],'max_tokens':5}}).encode(),"
        f"headers={{'Authorization':'Bearer {api_key}','Content-Type':'application/json'}});"
        "print(json.loads(urllib.request.urlopen(req,timeout=10).read())['choices'][0]['message']['content'])"
    )
    r = subprocess.run(
        ["docker", "exec", "agenta-oss-gh-services-1", "python", "-c", code],
        capture_output=True, text=True,
    )
    print(f"docker test ({docker_url})")
    if r.returncode == 0: print(f"  chat: {r.stdout.strip()}")
    else: print(f"  FAIL: {r.stderr.strip()}")


def serve():
    session = boto3.Session()
    region = session.region_name or "eu-central-1"
    bedrock = session.client("bedrock", region_name=region)
    response = bedrock.list_foundation_models(byOutputModality="TEXT")
    active_models = [
        m["modelId"]
        for m in response["modelSummaries"]
        if m.get("modelLifecycle", {}).get("status") == "ACTIVE"
    ]

    config = {
        "model_list": [
            {
                "model_name": "bedrock/*",
                "litellm_params": {
                    "model": "bedrock/*",
                    "aws_region_name": region,
                }
            }
        ],
        "general_settings": {
            "master_key": api_key
        }
    }

    with open("litellm_config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"region:     {region}")
    print(f"host url:   {base_url}")
    print(f"docker url: {docker_url}")
    print(f"api key:    {api_key}")
    print(f"models:     {len(active_models)} active")
    print(f"\ntest: uv run litellmproxy.py test|docker\n")
    run_server(
        args=["--config", "litellm_config.yaml", "--host", host, "--port", str(port)],
        standalone_mode=False,
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    {"serve": serve, "test": test, "docker": docker}[cmd]()
