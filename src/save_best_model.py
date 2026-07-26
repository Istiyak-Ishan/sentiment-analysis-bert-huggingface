from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from config import MODEL_DIR, MODEL_NAME


BEST_CHECKPOINT = "outputs/checkpoint-12630"


def save_model():

    # Load fine-tuned model weights
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_CHECKPOINT
    )

    # Load original BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Save final model package
    model.save_pretrained(MODEL_DIR)

    tokenizer.save_pretrained(MODEL_DIR)

    print("Model and tokenizer saved successfully!")


if __name__ == "__main__":
    save_model()