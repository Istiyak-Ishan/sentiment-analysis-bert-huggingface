from datasets import load_dataset
from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    DATASET_NAME,
    DATASET_CONFIG,
    MAX_LENGTH,
)


def load_sst2():
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized