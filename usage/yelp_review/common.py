from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_dataloader():
    dataset = load_dataset("yelp_review_full")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(64))

    dataloader = DataLoader(small_dataset, shuffle=False, batch_size=8)
    return dataloader
