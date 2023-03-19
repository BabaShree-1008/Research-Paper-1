import os
from datasets import load_dataset

dataset = load_dataset("sst2")

# save dataset to a file

splits = ["train", "validation", "test"]
MAX_SAMPLE_SIZE = 40000

for split in splits:
    print(split)
    text_data = []
    count = 0
    for sample in dataset[split]:
        print(count)
        data = sample["sentence"]
        text_data.append(data)
        if count >= MAX_SAMPLE_SIZE:
            break
        count += 1

    with open(f"{split}_text.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(text_data))