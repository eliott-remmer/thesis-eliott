import os

import numpy as np
from datasets import load_dataset, load_metric, set_caching_enabled
from decouple import config
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Logging
from transformers.integrations import NeptuneCallback
from transformers.trainer_utils import set_seed

BATCH_SIZE = config("BATCH_SIZE", cast=int, default=1)
MODEL_DIR = config("MODEL_DIR", default="/workspace/models")
DATA_DIR = config("DATA_DIR", default="/workspace/data")
CACHE_DIR = config("CACHE_DIR", default="/.cache/huggingface")
LOG_DIR = config("LOG_DIR", default="/workspace/logs")

# Neptune API (loaded as global env. variables from the Training Callback)
NEPTUNE_API_TOKEN = config("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = config("NEPTUNE_PROJECT")

# Huggingface transformers specific
MODEL_NAME = "bert-base-uncased"
METRIC_NAME = "f1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tokenize_fn(example):
    return tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=70
    )  # twitter max length 70, example['text'], #movies max length 5123, example['review']


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


if __name__ == "__main__":
    # Define model and data

    # DATASET = "movies"
    DATASET = "tweets"
    set_caching_enabled(True)
    set_seed(42)
    if DATASET == "movies":
        NUM_LABELS = 2
        dataset_raw = load_dataset("movie_rationales", cache_dir=CACHE_DIR)
    elif DATASET == "tweets":
        NUM_LABELS = 3
        dataset_raw = load_dataset(
            DATA_DIR + "/tweet-sentiment-extraction",
            data_files={"train": "train_for_fine_tune.csv", "test": "test_for_fine_tune.csv"},
        )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    metric = load_metric(METRIC_NAME)

    # Prepate the data
    dataset = dataset_raw.map(tokenize_fn, batched=True)
    data_collator = default_data_collator

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Train and evaluate the model
    training_args = TrainingArguments(
        f"{MODEL_DIR}/{MODEL_NAME}-finetuned-" + DATASET,
        per_device_train_batch_size=13,
        per_device_eval_batch_size=13,
        learning_rate=0.000009828,
        num_train_epochs=3,
        logging_dir=LOG_DIR,
        warmup_ratio=0.01295,
        weight_decay=0.01637,
        metric_for_best_model=METRIC_NAME,
        evaluation_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            NeptuneCallback,
            # PrinterCallback
            # TensorBoardCallback(SummaryWriter(log_dir=LOG_DIR)),
        ],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    result_train = trainer.train()
    result_eval = trainer.evaluate()
    trainer.save_model()
