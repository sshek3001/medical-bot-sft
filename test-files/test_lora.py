import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


# Local path to the base Llama model.
MODEL_PATH = r"D:\models\llama-3.2-1b-instruct"


# Load the base model in 4-bit precision.
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


# Define the LoRA adapter.
# r controls the size of the trainable LoRA matrices.
# lora_alpha scales their contribution.
# dropout randomly drops some LoRA activations during training.
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


# Attach the LoRA adapters to the base model.
model = get_peft_model(model, lora_config)


# Print the number of trainable vs total parameters.
model.print_trainable_parameters()