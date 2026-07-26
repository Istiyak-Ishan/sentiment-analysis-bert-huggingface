import torch
import pandas as pd

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from datasets import load_dataset

from config import MODEL_DIR, MAX_LENGTH


def load_model():
    """
    Load fine-tuned BERT model and tokenizer.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    return model, tokenizer



def prepare_dataset(tokenizer):
    """
    Load SST-2 validation dataset
    and tokenize it.
    """

    dataset = load_dataset(
        "glue",
        "sst2"
    )

    validation = dataset["validation"]

    # Keep original sentences for error analysis
    sentences = validation["sentence"]


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


    return validation, sentences



def analyze_errors():

    model, tokenizer = load_model()


    dataset, sentences = prepare_dataset(
        tokenizer
    )


    dataloader = DataLoader(
        dataset,
        batch_size=16
    )


    predictions = []
    labels = []
    confidences = []


    model.eval()


    index = 0


    with torch.no_grad():

        for batch in dataloader:


            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )


            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )


            batch_predictions = torch.argmax(
                probabilities,
                dim=1
            )


            batch_confidence = torch.max(
                probabilities,
                dim=1
            ).values



            for pred, label, confidence in zip(
                batch_predictions,
                batch["label"],
                batch_confidence
            ):

                predictions.append(
                    pred.item()
                )

                labels.append(
                    label.item()
                )

                confidences.append(
                    confidence.item()
                )


            index += len(batch["label"])



    results = pd.DataFrame(
        {
            "sentence": sentences,
            "actual": labels,
            "predicted": predictions,
            "confidence": confidences
        }
    )


    # Save all predictions
    results.to_csv(
        "all_predictions.csv",
        index=False
    )


    # Filter wrong predictions
    errors = results[
        results["actual"] != results["predicted"]
    ]


    errors.to_csv(
        "wrong_predictions.csv",
        index=False
    )


    print(
        f"Total samples: {len(results)}"
    )

    print(
        f"Total mistakes: {len(errors)}"
    )


    print("\nFirst 10 mistakes:\n")

    print(
        errors.head(10)
    )



if __name__ == "__main__":

    analyze_errors()