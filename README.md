# RAG Failure Study: When Retrieval Hurts Generation

This repository contains a small experimental study exploring how retrieval quality affects the reliability of Retrieval-Augmented Generation (RAG) systems.

RAG is commonly used to improve LLM answers by retrieving external documents and providing them as context. However, when retrieval introduces noisy or misleading information, it can negatively influence generation.

This experiment explores how different retrieval strategies behave when the retrieved context is imperfect.

The goal of this project is not to create a large benchmark, but to illustrate potential RAG failure modes in a controlled experimental setup.

---

# Experiment Overview

The experiment evaluates how different retrieval strategies affect answer correctness and failure behaviour.

A small question set is used together with a curated knowledge base and intentionally misleading distractor documents.

Four system configurations are compared:

**LLM_ONLY**  
The baseline model answers questions without retrieval.

**RAG_TOP1**  
The model retrieves the single highest-ranked document and uses it as context.

**RAG_TOP3**  
The model retrieves the top three documents and uses them as context.

**RAG_TOP3_GUARDED**  
The same retrieval setup as RAG_TOP3, but the model is allowed to abstain when the context appears unreliable.

The experiment measures:

- Answer correctness
- Grounding / faithfulness to retrieved context
- Failure types such as retrieval errors, grounding errors, and abstentions

---

# Dataset

The evaluation uses:

- 15 technical questions related to topics such as Transformers, BERT, RAG, FAISS, embeddings, and attention mechanisms.
- A small knowledge base containing documents with correct information.
- Additional distractor documents designed to simulate retrieval noise.

These distractor documents intentionally contain misleading or incorrect statements to test how RAG behaves under imperfect retrieval conditions.

---

# Evaluation Pipeline

The experiment follows these steps:

1. Load questions and document corpus
2. Build embeddings for documents
3. Use FAISS for similarity search
4. Retrieve top-k documents depending on the condition
5. Generate answers using the LLM
6. Automatically label outputs using rule-based evaluation
7. Analyze correctness and failure types
8. Generate visualizations summarizing the results

---

# Key Observations

In this small experimental setting:

- Top-1 retrieval sometimes reduced accuracy compared to the baseline LLM.
- Retrieving multiple documents partially recovered performance.
- Guardrails reduced incorrect answers but increased abstentions.

These results suggest that RAG performance can be sensitive to retrieval quality and context selection.

---

# Repository Structure


rag_failure_study/
│

├── docs/ # Knowledge base and distractor documents

├── questions.json # Question set used for evaluation

├── run_experiment.py # Runs the RAG experiment

├── auto_label.py # Automatically labels outputs

├── make_onepager.py # Generates summary visualizations

│

├── outputs_raw.csv # Raw model outputs

├── outputs_scored.csv # Evaluated results

├── one_pager.png # Final experiment summary figure

│

└── README.md


---

# Running the Experiment

Install dependencies:


pip install -r requirements.txt


Run the experiment:


python run_experiment.py


Score the outputs:


python auto_label.py


Generate the visualization:


python make_onepager.py


This will produce the final summary figure:


one_pager.png


---

# Limitations

This is a small exploratory study with a limited dataset (15 questions).  
The goal is to illustrate potential RAG failure modes rather than provide statistically strong conclusions.

Results may vary with different datasets, models, or retrieval systems.

---

# Future Work

Possible extensions include:

- Larger evaluation datasets
- Multiple LLMs
- Reranking strategies
- Retrieval filtering
- Improved guardrail mechanisms

---

# License

This project is intended for educational and experimental purposes.
