"""Utilities for loading and formatting the SciQ dataset."""

from datasets import load_dataset, Dataset


def load_sciq_dataset():
    """Load the SciQ dataset from Hugging Face."""
    print("Loading the SciQ dataset...")
    dataset = load_dataset("sciq")
    print(f"Number of training examples: {len(dataset['train'])}")
    print(f"Number of validation examples: {len(dataset['validation'])}")
    return dataset


def format_data(split):
    """Format dataset into a list of dictionaries with input and output keys."""
    formatted = []
    for ex in split:
        context = ex.get("support", "")
        formatted.append(
            {
                "input": f"Question: {ex['question']} Context: {context}",
                "output": ex["correct_answer"],
            }
        )
    return formatted


def prepare_datasets(dataset):
    """Convert raw splits into Hugging Face `Dataset` objects."""
    print("Formatting the data...")
    train_data = format_data(dataset["train"])
    valid_data = format_data(dataset["validation"])

    print(f"Number of formatted training examples: {len(train_data)}")
    print(f"Number of formatted validation examples: {len(valid_data)}")

    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(valid_data)
    return train_dataset, valid_dataset


if __name__ == "__main__":
    ds = load_sciq_dataset()
    train_ds, val_ds = prepare_datasets(ds)
    print("Sample formatted training example:")
    print(train_ds[0])
