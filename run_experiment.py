import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# -----------------------
# Config
# -----------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
DOCS_DIR = "docs"
QUESTIONS_FILE = "questions.json"
OUT_RAW = "outputs_raw.csv"

client = OpenAI()

def load_docs(folder: str):
    texts, names = [], []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".txt"):
            path = os.path.join(folder, fn)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())
            names.append(fn)
    if len(texts) == 0:
        raise ValueError(f"No .txt files found in {folder}/")
    return texts, names

def build_faiss_index(doc_texts, embedder):
    # cosine similarity via normalized vectors + inner product
    X = embedder.encode(doc_texts, normalize_embeddings=True).astype(np.float32)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    return index

def retrieve(question: str, k: int, embedder, index, doc_texts, doc_names):
    qv = embedder.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, idx = index.search(qv, k)
    picks = []
    for s, i in zip(scores[0], idx[0]):
        picks.append(
            {
                "doc": doc_names[int(i)],
                "score": float(s),
                "text": doc_texts[int(i)],
            }
        )
    return picks

def ask_llm(prompt: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content.strip()

def format_context(picks):
    return "\n\n".join([f"[{p['doc']}] {p['text']}" for p in picks])

def run_condition(question: str, condition: str, embedder, index, doc_texts, doc_names):
    if condition == "LLM_ONLY":
        ans = ask_llm(question)
        return ans, [], ""

    if condition in ("RAG_TOP1", "RAG_TOP3", "RAG_TOP3_GUARDED"):
        k = 1 if condition == "RAG_TOP1" else 3
        picks = retrieve(question, k, embedder, index, doc_texts, doc_names)
        ctx = format_context(picks)

        if condition == "RAG_TOP3_GUARDED":
            prompt = (
                "Answer ONLY using the provided context. "
                "If the context does not contain enough information to answer, say exactly:\n"
                "I don’t know from the given context.\n\n"
                "Include 1–2 citations in square brackets like [filename].\n\n"
                f"Context:\n{ctx}\n\nQuestion:\n{question}"
            )
        else:
            prompt = (
                "Use the context to answer the question.\n\n"
                f"Context:\n{ctx}\n\nQuestion:\n{question}"
            )

        ans = ask_llm(prompt)
        return ans, picks, ctx

    raise ValueError(f"Unknown condition: {condition}")

def main():
    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"Missing folder: {DOCS_DIR}/")
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(f"Missing file: {QUESTIONS_FILE}")

    doc_texts, doc_names = load_docs(DOCS_DIR)

    embedder = SentenceTransformer(EMB_MODEL)
    index = build_faiss_index(doc_texts, embedder)

    questions = json.load(open(QUESTIONS_FILE, "r", encoding="utf-8"))

    conditions = ["LLM_ONLY", "RAG_TOP1", "RAG_TOP3", "RAG_TOP3_GUARDED"]

    rows = []
    for q in tqdm(questions, desc="Running experiments"):
        qid = q.get("id", "")
        question = q["question"]

        for c in conditions:
            ans, picks, ctx = run_condition(question, c, embedder, index, doc_texts, doc_names)

            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "condition": c,
                    "answer": ans,
                    "retrieved_docs": ", ".join([p["doc"] for p in picks]),
                    "retrieved_scores": ", ".join([f"{p['score']:.4f}" for p in picks]),
                    "context": ctx,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RAW, index=False)
    print(f"\n✅ Saved: {OUT_RAW}")
    print("Next: create outputs_scored.csv using score_template.py")

if __name__ == "__main__":
    main()