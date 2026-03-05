import pandas as pd
import matplotlib.pyplot as plt

SCORED = "outputs_scored.csv"

df = pd.read_csv(SCORED)

df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
df["is_faithful"] = pd.to_numeric(df["is_faithful"], errors="coerce")

# Accuracy by condition
acc = df.groupby("condition")["is_correct"].mean()

plt.figure()
plt.bar(acc.index, acc.values)
plt.title("Accuracy by Condition")
plt.ylabel("Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("accuracy.png", dpi=200)

# Faithfulness by condition (RAG only)
rag = df[df["condition"] != "LLM_ONLY"].copy()
faith = rag.groupby("condition")["is_faithful"].mean()

plt.figure()
plt.bar(faith.index, faith.values)
plt.title("Faithfulness by RAG Condition")
plt.ylabel("Faithfulness")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("faithfulness.png", dpi=200)

# Failure types (RAG only)
ft = rag.groupby("condition")["failure_type"].value_counts(normalize=True).unstack(fill_value=0)

plt.figure()
ft.plot(kind="bar", stacked=True)
plt.title("RAG Failure Types (Proportion)")
plt.ylabel("Proportion")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("failure_types.png", dpi=200)

print("✅ Saved: accuracy.png, faithfulness.png, failure_types.png")