import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# Path to our local Llama model.
MODEL_PATH = r"D:\models\llama-3.2-1b-instruct"

# Load our conversational training data.
print("Starting train.py", flush=True)

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)

print("Dataset loaded", flush=True)

# Load the tokenizer associated with Llama.
print("Loading tokenizer", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Tokenizer loaded", flush=True)


# Load Llama in 4-bit precision.
# This is the "Q" part of QLoRA.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading 4-bit model", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded", flush=True)


# Configure the LoRA adapters.
# Only these adapter parameters will be trained.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)


# Configure SFT.
# For now this only defines how training should behave.
training_args = SFTConfig(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    report_to="none"
)


# Create the trainer.
print("Creating SFTTrainer", flush=True)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    peft_config=lora_config
)

print("SFTTrainer created successfully", flush=True)
trainer.train()