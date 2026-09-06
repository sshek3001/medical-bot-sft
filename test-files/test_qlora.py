import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# Local path to our base Llama model.
MODEL_PATH = r"D:\models\llama-3.2-1b-instruct"


# Tell bitsandbytes to load the model weights in 4-bit precision.
# NF4 is the quantization format commonly used for QLoRA.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# Load Llama using the 4-bit configuration.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)


# Print where the model is located.
print("Model loaded successfully.")
print("Model device:", next(model.parameters()).device)


# Print GPU memory currently allocated by PyTorch.
memory_used = torch.cuda.memory_allocated() / 1024**3
print(f"GPU memory allocated: {memory_used:.2f} GB")