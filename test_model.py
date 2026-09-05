import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = r"D:\models\llama-3.2-1b-instruct"


# ---------------------------------------------------------
# 1. Load the tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# ---------------------------------------------------------
# 2. Reset GPU memory statistics
# ---------------------------------------------------------
torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------
# 3. Load the model
# ---------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16
)


# ---------------------------------------------------------
# 4. Move the model to the GPU
# ---------------------------------------------------------
model = model.to("cuda")

print("Model loaded successfully!")
print("Running on:", next(model.parameters()).device)


# ---------------------------------------------------------
# 5. Display GPU memory used by the model
# ---------------------------------------------------------
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"GPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved:  {reserved:.2f} GB")


# ---------------------------------------------------------
# 6. Get user input
# ---------------------------------------------------------
user_input = input("\nYou: ")

messages = [
    {
        "role": "user",
        "content": user_input
    }
]


# ---------------------------------------------------------
# 7. Tokenize the input
# ---------------------------------------------------------
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True
).to("cuda")


# ---------------------------------------------------------
# 8. Generate a response
# ---------------------------------------------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )


# ---------------------------------------------------------
# 9. Display peak GPU memory used during inference
# ---------------------------------------------------------
peak_memory = torch.cuda.max_memory_allocated() / 1024**3

print(f"\nPeak GPU memory allocated: {peak_memory:.2f} GB")


# ---------------------------------------------------------
# 10. Decode the response
# ---------------------------------------------------------
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)
print("\nLlama:", response)