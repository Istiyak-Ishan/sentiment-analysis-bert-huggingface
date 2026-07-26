import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import MODEL_DIR


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        output_attentions=True
    )

    model.eval()

    return tokenizer, model



def visualize_attention(sentence):

    tokenizer, model = load_model()


    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    attentions = outputs.attentions


    # Last layer attention
    last_layer_attention = attentions[-1]


    # First attention head
    attention = last_layer_attention.mean(
    dim=1
    )[0]


    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )


    plt.figure(
        figsize=(10,8)
    )


    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis"
    )


    plt.title(
        "BERT Attention Visualization"
    )

    plt.xlabel(
        "Key Tokens"
    )

    plt.ylabel(
        "Query Tokens"
    )


    plt.xticks(
        rotation=45
    )

    plt.tight_layout()


    plt.savefig(
        "attention_heatmap.png"
    )


if __name__ == "__main__":

    sentence = (
        "The movie was absolutely amazing"
    )

    visualize_attention(sentence)