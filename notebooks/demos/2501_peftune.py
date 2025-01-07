"""Run fine-tuning on a model.

Main purpose: keep a combo (gemma-2b+peft+bsiz) that can fit on a T4.
C/p for a quick run.

Run:
    uv run src/tk/expt/gemmatune.py

"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "peft",
#   "transformers",
#   "datasets",
#   "torch",
# ]
# ///
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto")
dataset = load_dataset("squad")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def preprocess(examples):
    prompts = [
        f"Question: {q}\nContext: {c}\nAnswer: {a}"
        for q, c, a in zip(
            examples['question'],
            examples['context'],
            examples['answers'])
    ]
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset['train'].column_names,
    batched=True
)

training_args = TrainingArguments(
    output_dir="./gemma-qa",
    learning_rate=3e-4,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=500,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
)

trainer.train()
trainer.save_model("./gemma-qa-finetuned")