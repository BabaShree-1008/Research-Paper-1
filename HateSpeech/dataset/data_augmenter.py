from textattack.augmentation.recipes import EmbeddingAugmenter
import tqdm

DATA_PATH = "./benign"
AUG_DATA_PATH = "./augmented"

augmenter = EmbeddingAugmenter(
    pct_words_to_swap=1
)

with open(f"{DATA_PATH}/test_text.txt", "r") as test_data:
    test_data = test_data.readlines()

with open(f"{DATA_PATH}/val_text.txt", "r") as val_data:
    val_data = val_data.readlines()

with open(f"{DATA_PATH}/train_text.txt", "r") as train_data:
    train_data = train_data.readlines()

datasets = {}
datasets["train"] = str(train_data).split(sep="\\n")
datasets["test"] = str(test_data).split(sep="\\n")
datasets["val"] = str(val_data).split(sep="\\n")

for data in tqdm.tqdm(datasets):
    aug_data = []
    print(data)
    for sentence in tqdm.tqdm(datasets[data]):
        aug_string = augmenter.augment(sentence)
        aug_data.append(aug_string[0])
    with open(f"{AUG_DATA_PATH}/{data}.text") as file:
        file.write("\n".join(aug_data))