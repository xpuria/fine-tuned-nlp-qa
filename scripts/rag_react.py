"""Example RAG + ReAct pipeline using a fine-tuned T5 model."""

import json
from pathlib import Path

import faiss
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)


def load_index(index_dir: str):
    faiss_index = faiss.read_index(str(Path(index_dir) / "faiss_index.bin"))
    contexts = []
    with open(Path(index_dir) / "contexts.jsonl") as f:
        for line in f:
            contexts.append(json.loads(line)["text"])
    return faiss_index, contexts


def rag_react(question, model, tokenizer, q_model, q_tokenizer, faiss_index, contexts, device="cpu", top_k=3):
    q_inputs = q_tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        q_embedding = q_model(**q_inputs).pooler_output
        q_embedding = torch.nn.functional.normalize(q_embedding, p=2, dim=1)

    distances, indices = faiss_index.search(q_embedding.cpu().numpy(), top_k)
    retrieved = " ".join([contexts[i] for i in indices[0] if contexts[i].strip()])

    input_text = f"Question: {question} Context: {retrieved}"
    tokenized_inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(tokenized_inputs["input_ids"], max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(model_dir: str = "./results", index_dir: str = "./sciq_faiss_index"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

    faiss_index, contexts = load_index(index_dir)

    example_questions = [
        "What is the speed of light?",
        "Why is the sky blue?",
        "Explain Newton's first law of motion.",
    ]

    for q in example_questions:
        answer = rag_react(q, model, tokenizer, q_model, q_tokenizer, faiss_index, contexts, device, top_k=20)
        print(f"Q: {q}\nA: {answer}\n")


if __name__ == "__main__":
    main()
