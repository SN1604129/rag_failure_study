import pandas as pd

RAW = "outputs_raw.csv"
OUT = "outputs_scored.csv"

df = pd.read_csv(RAW)

# Add scoring columns (you will fill these)
df["is_correct"] = ""      # 1 or 0
df["is_faithful"] = ""     # 1 or 0 (for RAG conditions)
df["failure_type"] = ""    # none / retrieval_error / copy_error / grounding_failure / abstained

df.to_csv(OUT, index=False)
print(f"✅ Created: {OUT} (fill the 3 new columns)")