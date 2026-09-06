import json
from datasets import load_dataset


# Load the 100-example healthcare dataset from Hugging Face.
dataset = load_dataset(
    "davidberenstein1957/healthcare-customer-chat-sft"
)

# Get the training split.
records = dataset["train"]


# Create our own JSONL file.
# Each line will contain one conversation in the
# messages format expected by our SFT pipeline.
with open("data/train.jsonl", "w", encoding="utf-8") as f:

    for record in records:

        # Convert:
        # system_prompt + prompt + completion
        #
        # into:
        # system + user + assistant
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": record["system_prompt"]
                },
                {
                    "role": "user",
                    "content": record["prompt"]
                },
                {
                    "role": "assistant",
                    "content": record["completion"]
                }
            ]
        }

        # Write exactly one JSON object per line.
        f.write(
            json.dumps(conversation, ensure_ascii=False) + "\n"
        )


print(f"Converted {len(records)} records.")
print("Saved to: data/train.jsonl")