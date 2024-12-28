from transformers import (
    AutoTokenizer, FlaxAutoModelForCausalLM, TrainingArguments, FlaxTrainer,
    PreTrainedTokenizerBase,
)

from tk.models.gpt2 import GPT, GPTConfig
from tk.arc.expt.encoding import SimpleArcGridSeqEncoder


dataset, vocab = SimpleArcGridSeqEncoder.load(fmt='huggingface')
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model = GPT(config=GPTConfig(
    vocab_size=len(vocab),
    block_size=1024,
    num_heads=12,
    num_layers=12,
    use_bias=True,
    dtype='float32',
))


class PassThroughTokenizer(PreTrainedTokenizerBase):
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, text):
        return text

tokenizer = PassThroughTokenizer(vocab)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()
