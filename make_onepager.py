import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCORED = "outputs_scored.csv"
OUT = "one_pager.png"

df = pd.read_csv(SCORED)

# Ensure numeric
df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
df["is_faithful"] = pd.to_numeric(df["is_faithful"], errors="coerce")

conds_all = ["LLM_ONLY", "RAG_TOP1", "RAG_TOP3", "RAG_TOP3_GUARDED"]
conds_rag = ["RAG_TOP1", "RAG_TOP3", "RAG_TOP3_GUARDED"]

# --- Panel A: Dumbbell (accuracy shift vs LLM_ONLY)
acc = df.groupby("condition")["is_correct"].mean().reindex(conds_all)
base = float(acc["LLM_ONLY"])

# --- Panel B: Heatmap (per-question correctness)
pivot = df.pivot_table(index="qid", columns="condition", values="is_correct", aggfunc="mean").reindex(columns=conds_all)

# --- Panel C: Tradeoff (abstain vs wrong)
def rates(cond: str):
    sub = df[df["condition"] == cond]
    total = len(sub)
    abst = (sub["failure_type"] == "abstained").sum() / total if total else 0.0
    wrong = (sub["is_correct"] == 0).sum() / total if total else 0.0
    return abst, wrong

trade_abst, trade_wrong = [], []
for c in conds_rag:
    a, w = rates(c)
    trade_abst.append(a)
    trade_wrong.append(w)

# --- Panel D: Failure type distribution (stacked)
rag = df[df["condition"] != "LLM_ONLY"].copy()
ft = rag.groupby("condition")["failure_type"].value_counts(normalize=True).unstack(fill_value=0).reindex(conds_rag)

# Make it readable by keeping consistent order if present
ft_cols = [c for c in ["none", "retrieval_error", "copy_error", "grounding_failure", "abstained"] if c in ft.columns]
ft = ft[ft_cols]

# ---------- Plot layout ----------
fig = plt.figure(figsize=(14, 9))

# A) Dumbbell
ax1 = fig.add_subplot(2, 2, 1)
y = np.arange(len(conds_rag))
for i, c in enumerate(conds_rag):
    ax1.plot([base, acc[c]], [i, i], marker="o")
ax1.axvline(base, linestyle="--")
ax1.set_yticks(y)
ax1.set_yticklabels(conds_rag)
ax1.set_title("A) Accuracy shift vs LLM_ONLY")
ax1.set_xlabel("Accuracy")

# B) Heatmap
ax2 = fig.add_subplot(2, 2, 2)
im = ax2.imshow(pivot.values, aspect="auto")
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels(pivot.index)
ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels(pivot.columns, rotation=25, ha="right")
ax2.set_title("B) Per-question correctness (1=correct, 0=wrong)")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# C) Tradeoff scatter
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(trade_abst, trade_wrong)
for i, c in enumerate(conds_rag):
    ax3.text(trade_abst[i], trade_wrong[i], " " + c)
ax3.set_title("C) Safety tradeoff: Abstain vs Wrong")
ax3.set_xlabel("Abstain rate")
ax3.set_ylabel("Wrong-answer rate")

# D) Failure types stacked
ax4 = fig.add_subplot(2, 2, 4)
bottom = np.zeros(len(ft.index))
x = np.arange(len(ft.index))
for col in ft.columns:
    ax4.bar(x, ft[col].values, bottom=bottom, label=col)
    bottom += ft[col].values
ax4.set_xticks(x)
ax4.set_xticklabels(ft.index, rotation=25, ha="right")
ax4.set_title("D) RAG failure types (proportion)")
ax4.set_ylabel("Proportion")
ax4.legend(fontsize=8, loc="upper left")

fig.suptitle("RAG Failure Study — One-page Summary", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=220)
print(f"Saved: {OUT}")