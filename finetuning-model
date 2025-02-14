import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import torch
import os

df = pd.read_csv("unique_survey_questions2.csv")
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

model_name = "google/gemma-2b-it"  # Using Gemma 2B instruction-tuned
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    batch_size=4,
)

#  LoRA fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)

#  prompts for fine-tuning
def create_prompt(example):
    prompt = f"""### Instruction: Extract the domain, topic, and sentiment from the question below.

### Input: {example['Input']}

### Response: {example['Output']}"""
    return {"text": prompt}

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df).map(create_prompt)
val_dataset = Dataset.from_pandas(val_df).map(create_prompt)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = dict(
    output_dir="./gemma_finetuned_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    save_total_limit=3,
    optim="adamw_torch",
    bf16=True,  # Use bfloat16 for better performance
)

# Train the model
trainer = FastLanguageModel.get_trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

# Save fine-tuned model
save_path = "gemma_finetuned"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Convert to GGUF format using llama.cpp
os.system(f"python llama.cpp/convert-hf-to-gguf.py --model_dir {save_path} --outfile {save_path}.gguf")

print(f" Model successfully saved in GGUF format as {save_path}.gguf")

# Inference function
def generate_response(question, max_new_tokens=200):
    prompt = f"""### Instruction:
Analyze the question and extract domain, topic, and sentiment.

### Input:
{question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Testing the model
test_question = "How do Electric Vehicles impact mental health?"
print(generate_response(test_question))
