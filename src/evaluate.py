import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_DIR, MAX_LENGTH


def load_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    return model, tokenizer


def prepare_dataset(tokenizer):

    dataset = load_dataset(
        "glue",
        "sst2"
    )

    validation = dataset["validation"]

    def tokenize(batch):

        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    validation = validation.map(
        tokenize,
        batched=True
    )

    validation.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"
        ]
    )

    return validation


def evaluate():

    model, tokenizer = load_model()

    dataset = prepare_dataset(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=16
    )

    model.eval()

    predictions = []
    labels = []


    with torch.no_grad():

        for batch in dataloader:

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            logits = outputs.logits

            preds = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                batch["label"].cpu().numpy()
            )


    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "Negative",
                "Positive"
            ]
        )
    )


    cm = confusion_matrix(
        labels,
        predictions
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Negative",
            "Positive"
        ],
        yticklabels=[
            "Negative",
            "Positive"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        "BERT Sentiment Confusion Matrix"
    )

    plt.savefig(
        "confusion_matrix.png"
    )


if __name__ == "__main__":
    evaluate()