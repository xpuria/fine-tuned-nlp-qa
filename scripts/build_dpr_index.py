"""Build a FAISS index using DPR encoders for the SciQ dataset."""

import json
import gc

import faiss
import numpy as np
import torch
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)

from scripts.load_dataset import load_sciq_dataset


def encode_contexts(encoder, tokenizer, texts, batch_size=64, device="cpu"):
    """Encode texts in batches to manage GPU memory."""
    embeddings = []
    num_texts = len(texts)
    for start_idx in range(0, num_texts, batch_size):
        end_idx = min(start_idx + batch_size, num_texts)
        batch_texts = texts[start_idx:end_idx]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            batch_emb = encoder(**inputs).pooler_output
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            embeddings.append(batch_emb.cpu().numpy())
        print(f"Encoded contexts {start_idx} to {end_idx} (batch size {len(batch_texts)})")
    return np.vstack(embeddings)


def main(index_path: str = "./sciq_faiss_index"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_sciq_dataset()
    contexts = [ex.get("support", "") for ex in dataset["train"]]

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

    print("Encoding contexts...")
    context_embeddings = encode_contexts(q_model, q_tokenizer, contexts, batch_size=64, device=device)

    embedding_dim = context_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(context_embeddings)

    faiss.write_index(faiss_index, f"{index_path}/faiss_index.bin")
    with open(f"{index_path}/contexts.jsonl", "w") as f:
        for ctx in contexts:
            f.write(json.dumps({"text": ctx}) + "\n")
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
