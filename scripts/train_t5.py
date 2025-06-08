"""Fine-tune a T5 model on the formatted SciQ dataset."""

import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from scripts.load_dataset import load_sciq_dataset, prepare_datasets


def preprocess_data(tokenizer, examples):
    inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
    )
    labels = tokenizer(
        examples["output"],
        max_length=128,
        truncation=True,
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    }


def main(model_name: str = "t5-base", output_dir: str = "./results"):
    dataset = load_sciq_dataset()
    train_dataset, valid_dataset = prepare_datasets(dataset)

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    train_dataset = train_dataset.map(
        lambda x: preprocess_data(tokenizer, x), batched=True
    )
    valid_dataset = valid_dataset.map(
        lambda x: preprocess_data(tokenizer, x), batched=True
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
