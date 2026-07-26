from transformers import TrainingArguments, Trainer

from dataset import load_sst2, load_tokenizer, tokenize_dataset
from model import load_model
from metrics import compute_metrics
from config import (
    OUTPUT_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
)

dataset = load_sst2()

tokenizer = load_tokenizer()

tokenized_dataset = tokenize_dataset(dataset, tokenizer)

model = load_model()

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    fp16=True,

    logging_steps=100,

    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)
trainer.save_model(
    "saved_model"
)

tokenizer.save_pretrained(
    "saved_model"
)

trainer.train()