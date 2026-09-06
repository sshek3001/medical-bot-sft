from datasets import load_dataset

print("1. Script started", flush=True)

print("2. Loading dataset...", flush=True)

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)

print("3. Dataset loaded!", flush=True)
print(dataset)