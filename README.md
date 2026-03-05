# RAG Failure Study (1-Day Mini Research)

A small, research-style experiment to test **when Retrieval-Augmented Generation (RAG) helps and when it fails**.  
Instead of only reporting “RAG reduces hallucinations”, this repo introduces **controlled distractor documents** to measure how retrieval noise can *mislead* a model.

## Research Question
**Does RAG always improve factual accuracy, or can incorrect retrieval introduce new errors?**

## Hypotheses
- **H1:** RAG with **Top-1 retrieval** can be *worse than LLM-only* when retrieval returns a plausible but incorrect document.
- **H2:** **Top-3 retrieval** reduces this risk by adding redundancy (more chances to retrieve a correct source).
- **H3 (optional):** A **guardrail** (“cite evidence or say I don’t know”) reduces confident wrong answers.

## Experiment Conditions
We compare four conditions on the same question set:
1. **LLM_ONLY** — No retrieval
2. **RAG_TOP1** — Retrieve 1 document
3. **RAG_TOP3** — Retrieve 3 documents
4. **RAG_TOP3_GUARDED** — Retrieve 3 docs + “use context only; otherwise say I don’t know” + citations

## Controlled Knowledge Base
The `docs/` folder contains:
- **Correct documents** (ground truth)
- **Distractor documents** (plausible but wrong on one key fact)

This makes it possible to observe distinct failure modes:
- **Retrieval error**: retriever selects distractor(s)
- **Copy error**: model copies wrong distractor fact
- **Grounding failure**: correct docs are retrieved but model ignores/misuses them
- **Abstained**: guarded model refuses due to insufficient evidence

## Metrics
You will score outputs in `outputs_scored.csv`:
- **Accuracy** (`is_correct`): 1 if factually correct else 0
- **Faithfulness** (`is_faithful`): 1 if answer is supported by retrieved context (RAG conditions)
- **Failure type** (`failure_type`): `none | retrieval_error | copy_error | grounding_failure | abstained`

Charts generated:
- `accuracy.png`
- `faithfulness.png`
- `failure_types.png`

---

# Quickstart

## 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows

pip install -U openai faiss-cpu sentence-transformers pandas matplotlib tqdm python-dotenv
