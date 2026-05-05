# ============================================================
# 04b_validation_llm_judge.py
# LLM-as-judge validation of directed sentiment labels.
#
# Methodology:
#   - Sample rows where the pipeline assigned a non-neutral label
#     (these are the "evaluated" sentences - what we validate)
#   - Also sample a smaller set of neutral rows (to catch
#     false negatives where the pipeline missed real evaluations)
#   - For each sampled row, ask Claude to independently judge
#     the sentiment toward the target entity
#   - Compare to the pipeline label to compute precision/recall
#   - Stratify results by domestic vs foreign expert to test
#     the thesis premise that data loss is random with respect
#     to expert origin
#
# NOTE: Uses the Anthropic API. Set ANTHROPIC_API_KEY in your
# environment before running.
#
# Output:
#   validation_llm_judged_rows.csv   (per-row LLM decisions)
#   validation_llm_summary.txt       (precision/recall table)
# ============================================================

import os
import re
import time
import json
import random
from pathlib import Path

import pandas as pd
import anthropic
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

LONG_PATH   = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/mentions_directed_sentiment_long.csv"
OUT_ROWS    = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/validation_llm_judged_rows.csv"
OUT_SUMMARY = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/validation_llm_summary.txt"

# Sample sizes (adjust to control cost)
N_NONNEUTRAL_PER_STRATUM = 50   # per cell: (positive/negative) x (domestic/foreign) = 4 cells = 200 rows
N_NEUTRAL_SAMPLE         = 100  # neutral rows - for false-negative estimation

MODEL = "claude-haiku-4-5-20251001"  # fast and cheap; upgrade to sonnet for higher accuracy
TEMPERATURE = 0.0
MAX_TOKENS = 60   # only need a short label

RANDOM_SEED = 42

# ------------------------------------------------------------
# PROMPT TEMPLATE
# ------------------------------------------------------------

SYSTEM_PROMPT = """You are a political science research assistant helping to validate sentiment annotations on Russian news articles translated into English.

Your task: given a news sentence and an identified target entity, judge the evaluative sentiment expressed toward that entity.

Rules:
- Base your judgment ONLY on the provided text. Do not use outside knowledge.
- Focus on how the text frames the target entity - is the framing positive, negative, or neutral/factual?
- Quotes attributed to the expert count as the expert's evaluation.
- Answer with EXACTLY one word: Positive, Negative, or Neutral."""

USER_TEMPLATE = """Article sentence (translated from Russian):
\"\"\"
{sentence}
\"\"\"

Target entity: {target_fine}

Evaluative clause extracted by the pipeline:
\"\"\"
{clause}
\"\"\"

What is the sentiment expressed toward '{target_fine}' in this text?
Answer with exactly one word: Positive, Negative, or Neutral."""

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading pipeline output...")
long_df = pd.read_csv(LONG_PATH)
print(f"  {len(long_df):,} rows")

# Ensure we have the columns we need
needed = ["expert_id", "row_id", "sentence", "target_fine", "linked_clause_text",
          "sentiment_label", "is_foreign_final"]
missing = [c for c in needed if c not in long_df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

long_df["is_foreign_final"] = long_df["is_foreign_final"].map(
    lambda x: True if str(x).strip().lower() in ("true", "1", "yes") else False
)

# ------------------------------------------------------------
# STRATIFIED SAMPLE
# ------------------------------------------------------------

random.seed(RANDOM_SEED)

nonneutral = long_df[long_df["sentiment_label"].isin(["positive", "negative"])].copy()
neutral    = long_df[long_df["sentiment_label"] == "neutral"].copy()

sample_rows = []

# Non-neutral: stratify by (label) x (domestic/foreign)
for label in ["positive", "negative"]:
    for is_foreign in [True, False]:
        stratum = nonneutral[
            (nonneutral["sentiment_label"] == label) &
            (nonneutral["is_foreign_final"] == is_foreign)
        ]
        n = min(N_NONNEUTRAL_PER_STRATUM, len(stratum))
        if n == 0:
            continue
        sample_rows.append(stratum.sample(n, random_state=RANDOM_SEED))

# Neutral: stratify by domestic/foreign
for is_foreign in [True, False]:
    stratum = neutral[neutral["is_foreign_final"] == is_foreign]
    n = min(N_NEUTRAL_SAMPLE // 2, len(stratum))
    if n == 0:
        continue
    sample_rows.append(stratum.sample(n, random_state=RANDOM_SEED))

sample_df = pd.concat(sample_rows, ignore_index=True)
print(f"\nSample composition:")
print(sample_df.groupby(["sentiment_label", "is_foreign_final"]).size().to_string())
print(f"Total: {len(sample_df):,} rows")

# ------------------------------------------------------------
# LLM JUDGE
# ------------------------------------------------------------

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

def parse_llm_response(text: str) -> str:
    text_l = text.strip().lower()
    for label in ["positive", "negative", "neutral"]:
        if label in text_l:
            return label
    return "unknown"


def judge_row(sentence: str, target_fine: str, clause: str) -> dict:
    prompt = USER_TEMPLATE.format(
        sentence=sentence,
        target_fine=target_fine,
        clause=clause
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        label = parse_llm_response(raw)
        return {"llm_label": label, "llm_raw": raw, "llm_error": None}
    except Exception as e:
        return {"llm_label": "error", "llm_raw": None, "llm_error": str(e)}


print("\nRunning LLM judge...")
llm_results = []

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    result = judge_row(
        sentence   = str(row.get("sentence", "")),
        target_fine= str(row.get("target_fine", "")),
        clause     = str(row.get("linked_clause_text", ""))
    )
    llm_results.append(result)
    # Small pause to avoid rate-limiting
    time.sleep(0.1)

llm_df = pd.DataFrame(llm_results)
judged = pd.concat([sample_df.reset_index(drop=True), llm_df], axis=1)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

# For precision/recall we treat LLM label as ground truth.
# We focus on the non-neutral rows (the evaluated sentences).
# P/R for positive and negative classes separately.

def precision_recall_f1(y_true, y_pred, pos_class):
    tp = ((y_true == pos_class) & (y_pred == pos_class)).sum()
    fp = ((y_true != pos_class) & (y_pred == pos_class)).sum()
    fn = ((y_true == pos_class) & (y_pred != pos_class)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": prec, "recall": rec, "f1": f1}


valid = judged[judged["llm_label"].isin(["positive", "negative", "neutral"])].copy()
nonneutral_valid = valid[valid["sentiment_label"].isin(["positive", "negative"])]

lines = []
lines.append("=" * 65)
lines.append("LLM-AS-JUDGE VALIDATION RESULTS")
lines.append(f"Model used: {MODEL}")
lines.append("=" * 65)
lines.append(f"Total sampled rows: {len(sample_df):,}")
lines.append(f"  Non-neutral (pipeline): {len(nonneutral_valid):,}")
lines.append(f"  Neutral (pipeline):     {len(valid[valid['sentiment_label'] == 'neutral']):,}")
lines.append(f"Rows with valid LLM label: {len(valid):,}")
lines.append(f"LLM errors / unparseable: {(judged['llm_label'] == 'error').sum() + (judged['llm_label'] == 'unknown').sum()}")
lines.append("")

lines.append("--- LLM label distribution (non-neutral pipeline rows) ---")
lines.append(nonneutral_valid["llm_label"].value_counts().to_string())
lines.append("")

lines.append("--- Agreement rate (pipeline vs LLM) ---")
agreement = (valid["sentiment_label"] == valid["llm_label"]).mean()
lines.append(f"  Overall agreement: {100*agreement:.1f}%")

for fg_val, fg_label in [(True, "Foreign"), (False, "Domestic")]:
    sub = valid[valid["is_foreign_final"] == fg_val]
    agr = (sub["sentiment_label"] == sub["llm_label"]).mean()
    lines.append(f"  {fg_label}: {100*agr:.1f}%  (n={len(sub)})")

lines.append("")
lines.append("--- Precision / Recall (pipeline label, LLM as ground truth) ---")
lines.append("  (non-neutral rows only)")
lines.append("")

for cls in ["positive", "negative"]:
    m = precision_recall_f1(
        nonneutral_valid["llm_label"],
        nonneutral_valid["sentiment_label"],
        cls
    )
    lines.append(f"  Class: {cls.upper()}")
    lines.append(f"    Precision: {m['precision']:.3f}  |  Recall: {m['recall']:.3f}  |  F1: {m['f1']:.3f}")
    lines.append(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    lines.append("")

lines.append("--- False negative estimate (neutral rows) ---")
neutral_valid = valid[valid["sentiment_label"] == "neutral"]
fn_est = (neutral_valid["llm_label"] != "neutral").mean()
lines.append(f"  Neutral rows where LLM disagrees (possible false negatives): "
             f"{100*fn_est:.1f}%  (n={len(neutral_valid)})")

for fg_val, fg_label in [(True, "Foreign"), (False, "Domestic")]:
    sub = neutral_valid[neutral_valid["is_foreign_final"] == fg_val]
    fn = (sub["llm_label"] != "neutral").mean()
    lines.append(f"    {fg_label}: {100*fn:.1f}%  (n={len(sub)})")

lines.append("")
lines.append("--- Confusion matrix (pipeline rows, LLM as GT) ---")
lines.append("  (non-neutral pipeline rows only)")
ct = pd.crosstab(
    nonneutral_valid["llm_label"].rename("LLM (GT)"),
    nonneutral_valid["sentiment_label"].rename("Pipeline"),
    margins=True
)
lines.append(ct.to_string())
lines.append("")
lines.append("--- Domestic vs Foreign error rate comparison ---")
lines.append("  (supports thesis premise that data loss is non-systematic)")
for fg_val, fg_label in [(True, "Foreign"), (False, "Domestic")]:
    sub = nonneutral_valid[nonneutral_valid["is_foreign_final"] == fg_val]
    wrong = (sub["sentiment_label"] != sub["llm_label"]).sum()
    pct = 100 * wrong / max(len(sub), 1)
    lines.append(f"  {fg_label}: {wrong} / {len(sub)} errors  ({pct:.1f}%)")

summary_text = "\n".join(lines)
print("\n" + summary_text)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

Path(OUT_ROWS).parent.mkdir(parents=True, exist_ok=True)
judged.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
Path(OUT_SUMMARY).write_text(summary_text, encoding="utf-8")

print(f"\nRow-level results: {OUT_ROWS}")
print(f"Summary:           {OUT_SUMMARY}")
