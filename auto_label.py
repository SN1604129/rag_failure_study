import re
import pandas as pd

RAW = "outputs_raw.csv"
OUT = "outputs_scored.csv"

# -----------------------
# Gold rules for correctness (by qid)
# Each rule is a function(answer_lower)-> bool
# -----------------------
def has_all(text: str, *subs: str) -> bool:
    return all(s in text for s in subs)

def any_of(text: str, subs) -> bool:
    return any(s in text for s in subs)

def year_in(text: str, y: str) -> bool:
    return re.search(rf"\b{re.escape(y)}\b", text) is not None

GOLD = {
    "q1": lambda a: year_in(a, "2017"),
    "q2": lambda a: ("vaswani" in a) and year_in(a, "2017"),
    "q3": lambda a: has_all(a, "bidirectional", "encoder", "representations", "transformers"),
    "q4": lambda a: ("devlin" in a or "google" in a) and year_in(a, "2018"),
    "q5": lambda a: ("retrieval" in a) and ("context" in a or "documents" in a) and ("generation" in a),
    "q6": lambda a: any_of(a, ["yes", "uses", "provide", "retrieved"]) and any_of(a, ["documents", "external", "context"]),
    "q7": lambda a: any_of(a, ["focus", "relevant", "parts"]) and any_of(a, ["input", "sequence", "dependencies"]),
    "q8": lambda a: any_of(a, ["similarity", "nearest", "search", "vector"]) and any_of(a, ["faiss", "library"]) ,
    "q9": lambda a: any_of(a, ["semantic", "similarity", "vector"]) and any_of(a, ["retrieval", "search"]),
    "q10": lambda a: any_of(a, ["re-order", "rerank", "reorder"]) and any_of(a, ["retrieved", "documents", "results"]),
    "q11": lambda a: ("retrieval" in a) and ("generation" in a) and any_of(a, ["context", "documents"]),
    "q12": lambda a: any_of(a, ["precision", "recall", "noise"]) and any_of(a, ["small", "smaller", "large", "larger", "chunks"]),
    "q13": lambda a: any_of(a, ["noise", "wrong", "irrelevant", "mislead"]) and any_of(a, ["top-1", "top 1", "single"]),
    "q14": lambda a: any_of(a, ["redundancy", "multiple", "more", "evidence"]) and any_of(a, ["top-3", "top 3"]),
    "q15": lambda a: any_of(a, ["noise", "irrelevant", "wrong", "mislead"]) and any_of(a, ["hurts", "degrades", "worse", "errors"]),
}

# -----------------------
# Faithfulness: is key info supported by context?
# For this controlled study, we approximate faithfulness as:
# - if answer states the key gold fact(s), they must appear in context too.
# If the answer says "I don’t know from the given context." => faithful=1, correct depends (we'll treat as correct=0, faithful=1)
# -----------------------
KEY_FACTS = {
    "q1": ["2017"],
    "q2": ["vaswani", "2017"],
    "q3": ["bidirectional", "encoder", "representations", "transformers"],
    "q4": ["devlin", "2018"],
    "q5": ["retrieval", "context"],
    "q8": ["similarity", "search"],
    "q10": ["rerank", "re-order"],
    "q12": ["precision", "recall"],
}

DISTRACTOR_SIGNS = {
    "transformer_distractor.txt": ["2018"],
    "bert_distractor.txt": ["bayesian", "2019"],
    "rag_distractor.txt": ["fine-tunes", "without using", "no external"],
    "attention_distractor.txt": ["last token"],
    "faiss_distractor.txt": ["generate embeddings"],
    "embeddings_distractor.txt": ["hashes", "exact words only"],
    "rerank_distractor.txt": ["randomly", "shuffles"],
    "chunking_distractor.txt": ["chunking makes retrieval unnecessary"],
}

def context_contains_all(ctx: str, facts) -> bool:
    ctx = (ctx or "").lower()
    return all(f.lower() in ctx for f in facts)

def answer_mentions_any(ans: str, facts) -> bool:
    ans = (ans or "").lower()
    return any(f.lower() in ans for f in facts)

def main():
    df = pd.read_csv(RAW)

    is_correct = []
    is_faithful = []
    failure_type = []

    for _, row in df.iterrows():
        qid = str(row.get("qid", "")).strip()
        cond = str(row.get("condition", "")).strip()
        ans = str(row.get("answer", "")).strip()
        ans_l = ans.lower()
        ctx = str(row.get("context", "")).strip()
        ctx_l = ctx.lower()
        retrieved_docs = str(row.get("retrieved_docs", "")).strip()

        abstained = ans.strip() == "I don’t know from the given context."

        # correctness
        if abstained:
            corr = 0  # abstention is not an answer; keeps metric honest
        else:
            rule = GOLD.get(qid, None)
            corr = int(rule(ans_l)) if rule else 0

        # faithfulness
        if cond == "LLM_ONLY":
            faithful = ""  # N/A
        else:
            if abstained:
                faithful = 1
            else:
                key = KEY_FACTS.get(qid, None)
                if not key:
                    # fallback: if answer claims something but context is empty -> not faithful
                    faithful = 1 if len(ctx_l) > 0 else 0
                else:
                    # if answer mentions key facts, require context to contain them
                    if answer_mentions_any(ans_l, key):
                        faithful = int(context_contains_all(ctx_l, key))
                    else:
                        # if answer doesn't state key facts explicitly, treat as not-faithful
                        faithful = 0

        # failure type
        if cond == "LLM_ONLY":
            ft = "none" if corr == 1 else "none"
        else:
            if abstained:
                ft = "abstained"
            elif corr == 1:
                ft = "none"
            else:
                # retrieval_error: distractor doc retrieved
                got_distractor = False
                for d, signs in DISTRACTOR_SIGNS.items():
                    if d in retrieved_docs:
                        got_distractor = True
                        # copy_error if answer repeats distractor signature text
                        if any(s.lower() in ans_l for s in signs):
                            ft = "copy_error"
                            break
                else:
                    ft = "retrieval_error" if got_distractor else "grounding_failure"

        is_correct.append(corr)
        is_faithful.append(faithful)
        failure_type.append(ft)

    df["is_correct"] = is_correct
    df["is_faithful"] = is_faithful
    df["failure_type"] = failure_type

    df.to_csv(OUT, index=False)
    print(f"✅ Saved: {OUT}")
    print(df.groupby("condition")["is_correct"].mean())

if __name__ == "__main__":
    main()