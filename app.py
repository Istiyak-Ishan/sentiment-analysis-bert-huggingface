import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.config import MODEL_DIR, MAX_LENGTH


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    return tokenizer, model



tokenizer, model = load_model()



# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentiment(text):

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



# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="BERT Sentiment Classifier",
    page_icon="🎬"
)


st.title(
    "🎬 BERT Sentiment Classifier"
)


st.write(
    """
    This application uses a fine-tuned BERT model
    trained on the SST-2 movie review dataset.
    """
)


text = st.text_area(
    "Enter a movie review:",
    height=150,
    placeholder="Example: The movie was absolutely amazing!"
)



if st.button("Analyze Sentiment"):


    if text.strip():


        sentiment, confidence = predict_sentiment(
            text
        )


        st.subheader(
            "Prediction"
        )


        if sentiment == "Positive":

            st.success(
                f"😊 {sentiment}"
            )

        else:

            st.error(
                f"😞 {sentiment}"
            )


        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


    else:

        st.warning(
            "Please enter a review."
        )