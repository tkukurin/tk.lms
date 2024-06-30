
# LLM experiments

Just some LLM experiments.

Install in editable mode:

```bash
python -m venv venv
source venv/bin/activate.fish
pip install -e .
```

Install model
```
model='https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true'
out='data/phi3_fp4.gguf'
mkdir -p data
wget ${model} -O ${out}
```