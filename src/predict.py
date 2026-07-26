import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import MODEL_DIR, MAX_LENGTH



def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    return tokenizer, model



def predict_sentiment(text):

    tokenizer, model = load_model()


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()


    confidence = torch.max(
        probabilities
    ).item()


    labels = {
        0: "Negative",
        1: "Positive"
    }


    return (
        labels[prediction],
        confidence
    )



if __name__ == "__main__":


    text = input(
        "Enter movie review: "
    )


    sentiment, confidence = predict_sentiment(
        text
    )


    print("\nResult")
    print("----------------")
    print(
        f"Sentiment: {sentiment}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )