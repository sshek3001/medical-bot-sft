import json

INPUT_FILE = "data/train.jsonl"
OUTPUT_FILE = "data/train_converted.jsonl"


# Read the entire file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()


# The file contains multiple JSON objects.
# Decode them one after another.
decoder = json.JSONDecoder()
records = []

position = 0

while position < len(content):
    # Skip whitespace and blank lines
    while position < len(content) and content[position].isspace():
        position += 1

    if position >= len(content):
        break

    record, end_position = decoder.raw_decode(content, position)
    records.append(record)

    position = end_position


# Write each record as one JSON line
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"Converted {len(records)} records.")
print(f"Saved to: {OUTPUT_FILE}")