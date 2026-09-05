import json

FILE = "data/train.jsonl"

with open(FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        data = json.loads(line)

        print(f"Record {i}:")
        print("Messages:", len(data["messages"]))
        print("First role:", data["messages"][0]["role"])
        print()