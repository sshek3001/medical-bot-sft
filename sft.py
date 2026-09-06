import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# The original Llama model. We do NOT modify this model.
MODEL_PATH = r"D:\models\llama-3.2-1b-instruct"

# The LoRA adapter produced by our final training checkpoint.
ADAPTER_PATH = r"D:\SFT01\outputs\checkpoint-39"


# Load the tokenizer from the base model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# Load the base model in 4-bit, exactly as we did during training.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)


# Load our trained LoRA adapter on top of the frozen base model.
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)


print("Fine-tuned model loaded successfully.")


# Get a question to test the model.
user_input = input("\nYou: ")


# Format the conversation using Llama's chat template.
messages = [
    {
        "role": "user",
        "content": user_input
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True
).to("cuda")


# Generate the model's response.
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )


# Decode only the newly generated tokens.
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\nLlama:", response)