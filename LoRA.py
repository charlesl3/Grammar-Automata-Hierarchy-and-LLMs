# ================================================================
# GPT-2 + LoRA Fine-Tuning
# ================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model


# ------------------------------------------------
# 1. Load tokenizer
# ------------------------------------------------

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


# ------------------------------------------------
# 2. Load model
# ------------------------------------------------
# pad_token_id must be set for training with padding
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=tokenizer.eos_token_id,
)


# ------------------------------------------------
# 3. Apply LoRA to GPT-2
# ------------------------------------------------
# GPT-2 uses "c_attn" and "c_proj" for its QKV + projection layers.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # correct for GPT-2
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("✔ LoRA injected. Trainable params:", model.print_trainable_parameters())


# ------------------------------------------------
# 4. Tiny example dataset (you will replace this later)
# ------------------------------------------------
examples = [
    {"text": "I love learning about transformers and machine intelligence."},
    {"text": "The weather today is wonderful and the sun is shining."},
    {"text": "Natural language processing is incredibly powerful."},
]

dataset = Dataset.from_list(examples)


# ------------------------------------------------
# 5. Tokenization function
# ------------------------------------------------
# MUST include "labels" for Trainer to compute loss
def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    enc["labels"] = enc["input_ids"].copy()  # REQUIRED for GPT training
    return enc


tokenized_dataset = dataset.map(tokenize)


# ------------------------------------------------
# 6. TrainingArguments
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora-gpt2-out",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=1,
    fp16=False,                # MPS/CPU cannot use float16
    report_to="none",
)


# ------------------------------------------------
# 7. Trainer
# ------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)


# ------------------------------------------------
# 8. TRAIN
# ------------------------------------------------
trainer.train()

print("Training complete!")
print("Your LoRA fine-tuned GPT-2 is saved at: ./lora-gpt2-out")
