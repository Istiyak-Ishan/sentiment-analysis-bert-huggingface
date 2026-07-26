from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_DIR, MODEL_NAME


def save_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    save_model()

    print("Model saved successfully!")