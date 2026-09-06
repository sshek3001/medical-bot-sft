import json
import torch
from transformers import AutoTokenizer


# This points to the Llama tokenizer stored on your D: drive.
MODEL_PATH = r"D:\models\llama-3.2-1b-instruct"

# This is our training dataset.
INPUT_FILE = "data/train.jsonl"


# Load the tokenizer that belongs to our Llama model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# Open train.jsonl.
# Each line is one complete training conversation.
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]


# We will store the tokenized conversations here.
tokenized_records = []


# Process one conversation at a time.
for record in records:

    # Extract the list of system/user/assistant messages.
    messages = record["messages"]

    # Convert the entire conversation into the exact format
    # expected by Llama, then convert that text into token IDs.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    full_tokens = tokenizer(
        full_text,
        return_tensors="pt"
    )

    # Remove the batch dimension.
    # Shape changes from [1, sequence_length] to [sequence_length].
    input_ids = full_tokens["input_ids"][0]

    # Start by marking EVERY token with -100.
    # -100 tells CrossEntropyLoss to ignore that position.
    labels = torch.full_like(input_ids, -100)

    # We now find the positions belonging to each assistant message.
    for i, message in enumerate(messages):

        # We only want assistant messages to contribute to the loss.
        if message["role"] != "assistant":
            continue

        # Tokenize everything BEFORE this assistant message.
        # The length of this sequence tells us where the assistant
        # message begins in the full conversation.
        before_text = tokenizer.apply_chat_template(
            messages[:i],
            tokenize=False,
            add_generation_prompt=False
        )

        before_tokens = tokenizer(
            before_text,
            return_tensors="pt"
        )["input_ids"][0]

        # Tokenize everything INCLUDING this assistant message.
        # The resulting length tells us where the assistant
        # message ends in the full conversation.
        through_text = tokenizer.apply_chat_template(
            messages[:i + 1],
            tokenize=False,
            add_generation_prompt=False
        )

        through_tokens = tokenizer(
            through_text,
            return_tensors="pt"
        )["input_ids"][0]

        # The assistant message occupies this range of the
        # complete input sequence.
        start = len(before_tokens)
        end = len(through_tokens)

        # Copy the actual token IDs into the corresponding label positions.
        # All other positions remain -100.
        labels[start:end] = input_ids[start:end]

    # Save the input tokens and labels for this conversation.
    tokenized_records.append({
        "input_ids": input_ids,
        "labels": labels
    })


# Print a summary so we can verify that masking worked.
print(f"Loaded {len(tokenized_records)} conversations.")

for i, record in enumerate(tokenized_records):

    input_ids = record["input_ids"]
    labels = record["labels"]

    # Count the tokens that are NOT -100.
    # These are the tokens that will contribute to the training loss.
    training_tokens = (labels != -100).sum().item()

    print(
        f"Conversation {i + 1}: "
        f"{len(input_ids)} total tokens, "
        f"{training_tokens} training tokens"
    )