from transformers import AutoModelForSequenceClassification

from config import MODEL_NAME


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    return model