"""Evaluate a fine-tuned model using Hugging Face's Trainer."""

from transformers import pipeline


def main(model_dir: str = "./results"):
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model_dir,
        tokenizer=model_dir,
    )

    questions = [
        "What is the speed of light?",
        "Why is the sky blue?",
        "Explain Newton's first law of motion.",
    ]

    for q in questions:
        result = qa_pipeline(q)[0]["generated_text"]
        print(f"Q: {q}\nA: {result}\n")


if __name__ == "__main__":
    main()
