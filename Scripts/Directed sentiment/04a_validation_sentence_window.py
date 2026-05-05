# ============================================================
# 04a_validation_sentence_window.py
# Validate the single-sentence strategy:
#   (1) Data loss: how many neutral-coded rows have evaluative
#       content in the next 1-2 sentences?
#   (2) Noise: how often would including more sentences flip
#       the direction of already-coded non-neutral rows?
#
# Requires:
#   mentions_directed_sentiment_long.csv   (pipeline output)
#   article_level_clean.csv               (full article text)
#
# Output:
#   validation_sentence_window_results.csv   (per-row decisions)
#   validation_sentence_window_summary.txt   (headline numbers)
# ============================================================

import re
import json
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

LONG_PATH   = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/mentions_directed_sentiment_long.csv"
ARTICLE_PATH = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/Topic modelling/article_level_clean.csv"
OUT_ROW_PATH = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/validation_sentence_window_results.csv"
OUT_SUM_PATH = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/validation_sentence_window_summary.txt"

# How many sentences AFTER the mention sentence to inspect
NEXT_SENT_WINDOWS = [1, 2]   # produces separate estimates for +1 and +2

# ------------------------------------------------------------
# LEXICAL RESOURCES (copied from main script)
# ------------------------------------------------------------

POSITIVE_EVAL_WORDS = {
    "good", "better", "best", "strong", "stronger", "successful", "success",
    "effective", "efficient", "stabilized", "stability", "resilient", "resilience",
    "improved", "improving", "improvement", "advantage", "benefit", "beneficial",
    "constructive", "legitimate", "credible", "rational", "responsible",
    "competent", "confident", "safe", "peaceful", "supportive", "support",
    "victory", "win", "won", "gains", "gain", "promising", "useful", "necessary",
    "secure", "secured", "progress", "robust", "sound", "reasonable", "favorable",
    "favourable", "positive", "coherent", "calm", "steady", "sovereignty",
    "security", "partnership", "cooperation", "dialogue", "agreement", "multipolar",
    "independent", "balanced", "fair", "helpful", "normal", "stabilize", "recovery"
}

NEGATIVE_EVAL_WORDS = {
    "bad", "worse", "worst", "weak", "weaker", "failure", "failed", "failing",
    "ineffective", "inefficient", "collapse", "collapsed", "crisis", "chaos",
    "chaotic", "illegal", "dangerous", "threat", "threaten", "threatens",
    "mistake", "mistakes", "aggression", "aggressive", "violation", "violations",
    "corrupt", "corruption", "unstable", "instability", "irrational", "reckless",
    "irresponsible", "crime", "criminal", "terror", "terrorist", "violent",
    "violence", "hostile", "hostility", "problem", "problems", "harm", "harmful",
    "loss", "losses", "defeat", "defeated", "decline", "declining", "disaster",
    "catastrophe", "sanctions", "unjustified", "unacceptable", "provocation",
    "provocative", "escalation", "escalate", "escalated", "destabilizing",
    "destabilized", "fragile", "false", "lie", "lies", "hypocrisy",
    "interference", "intervention", "pressure", "biased", "illegitimate",
    "unlawful", "anti-russian", "unfair", "surround", "strangle", "ridiculous",
    "ridicule", "undermine", "undermined", "undermining", "ridiculed",
    "degrade", "degraded", "threatening", "isolation"
}

NEGATIONS = {"not", "no", "never", "none", "neither", "hardly", "scarcely", "without"}

TARGET_PATTERNS = {
    "Russia": ["russia", "russian federation", "kremlin", "moscow", "putin"],
    "Ukraine": ["ukraine", "ukrainian", "kyiv", "kiev", "zelensky", "zelenskyy"],
    "United States": ["united states", "u.s.", "usa", "america", "american", "washington", "white house"],
    "EU": ["european union", "eu", "brussels", "european commission", "european parliament"],
    "NATO": ["nato", "north atlantic alliance", "alliance"],
    "United Kingdom": ["united kingdom", "uk", "britain", "british", "london", "england"],
    "Europe": ["europe", "european", "europeans"],
    "West": ["the west", "west", "western", "collective west", "western countries", "western allies"]
}

# Build regex patterns for target detection
compiled_target_patterns = {}
for label, phrases in TARGET_PATTERNS.items():
    compiled_target_patterns[label] = [
        re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in phrases
    ]


def tokenize_simple(text: str):
    return re.findall(r"\b[\w'-]+\b", str(text).lower())


def lexical_eval_score(text: str) -> int:
    toks = tokenize_simple(text)
    score = 0
    for i, tok in enumerate(toks):
        prev = toks[max(0, i - 3): i]
        negated = any(t in NEGATIONS for t in prev)
        if tok in POSITIVE_EVAL_WORDS:
            score += -1 if negated else 1
        elif tok in NEGATIVE_EVAL_WORDS:
            score += 1 if negated else -1
    return score


def target_present_in_text(text: str, target_fine: str) -> bool:
    patterns = compiled_target_patterns.get(target_fine, [])
    for p in patterns:
        if p.search(text):
            return True
    return False


def lex_to_label(score: int) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"

# ------------------------------------------------------------
# SENTENCE SPLITTER
# Simple regex-based splitter (no spacy to keep it fast)
# ------------------------------------------------------------

_SENT_SPLIT = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
)


def split_sentences(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def find_mention_sentence_index(sentences: list, mention_sentence: str) -> int:
    """Return index of best-matching sentence (longest common substring match)."""
    if not sentences or not mention_sentence:
        return -1

    mention_norm = mention_sentence.strip().lower()

    # Exact match first
    for i, s in enumerate(sentences):
        if s.strip().lower() == mention_norm:
            return i

    # Substring match: find sentence that contains most of the mention
    best_idx = -1
    best_overlap = 0
    mention_words = set(tokenize_simple(mention_norm))
    if not mention_words:
        return -1

    for i, s in enumerate(sentences):
        s_words = set(tokenize_simple(s.lower()))
        overlap = len(mention_words & s_words) / len(mention_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    # Only accept if overlap is substantial
    if best_overlap >= 0.6:
        return best_idx
    return -1

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading pipeline output...")
long_df = pd.read_csv(LONG_PATH)
print(f"  Long file: {len(long_df):,} rows")

print("Loading article texts...")
art_df = pd.read_csv(ARTICLE_PATH, usecols=["row_id", "text_en"])
art_df = art_df.dropna(subset=["text_en"])
print(f"  Articles: {len(art_df):,} rows")

# Build row_id -> text_en lookup
art_lookup = dict(zip(art_df["row_id"], art_df["text_en"]))

# ------------------------------------------------------------
# EXTRACT NEXT SENTENCES
# ------------------------------------------------------------

print("Extracting next-sentence context for each pipeline row...")

results = []

for _, row in tqdm(long_df.iterrows(), total=len(long_df)):
    row_id      = row["row_id"]
    mention_sent = str(row.get("sentence", ""))
    target_fine  = row.get("target_fine", "")
    pipeline_label = row.get("sentiment_label", "neutral")
    is_foreign   = row.get("is_foreign_final", None)
    expert_id    = row.get("expert_id", None)

    article_text = art_lookup.get(row_id)
    rec = {
        "expert_id": expert_id,
        "row_id": row_id,
        "target_fine": target_fine,
        "pipeline_label": pipeline_label,
        "is_foreign_final": is_foreign,
        "mention_sentence": mention_sent,
        "article_found": article_text is not None,
    }

    if article_text is None:
        rec["mention_sent_idx"] = -1
        rec["n_article_sentences"] = 0
        for w in NEXT_SENT_WINDOWS:
            rec[f"next{w}_text"] = None
            rec[f"next{w}_has_target"] = False
            rec[f"next{w}_lex_score"] = 0
            rec[f"next{w}_lex_label"] = "neutral"
            rec[f"next{w}_would_change_neutral"] = False
            rec[f"next{w}_would_flip_nonneutral"] = False
        results.append(rec)
        continue

    sentences = split_sentences(article_text)
    idx = find_mention_sentence_index(sentences, mention_sent)

    rec["mention_sent_idx"] = idx
    rec["n_article_sentences"] = len(sentences)

    for w in NEXT_SENT_WINDOWS:
        # Collect the w sentences immediately after the mention sentence
        if idx < 0 or idx + w >= len(sentences):
            next_text = None
        else:
            next_sents = sentences[idx + 1 : idx + 1 + w]
            next_text = " ".join(next_sents)

        if not next_text:
            rec[f"next{w}_text"] = None
            rec[f"next{w}_has_target"] = False
            rec[f"next{w}_lex_score"] = 0
            rec[f"next{w}_lex_label"] = "neutral"
            rec[f"next{w}_would_change_neutral"] = False
            rec[f"next{w}_would_flip_nonneutral"] = False
            continue

        has_target = target_present_in_text(next_text, target_fine)
        lex_score  = lexical_eval_score(next_text) if has_target else 0
        lex_label  = lex_to_label(lex_score)

        # Combined lex score: mention sentence + next window
        mention_lex = lexical_eval_score(mention_sent)
        combined_lex = mention_lex + lex_score
        combined_label = lex_to_label(combined_lex)

        # Data loss: pipeline says neutral, but next sentences have evaluative signal toward target
        would_change_neutral = (
            pipeline_label == "neutral"
            and has_target
            and lex_label != "neutral"
        )

        # Noise: pipeline says non-neutral, but adding next sentences flips direction
        would_flip_nonneutral = (
            pipeline_label != "neutral"
            and combined_label != "neutral"
            and combined_label != pipeline_label
        )

        rec[f"next{w}_text"] = next_text
        rec[f"next{w}_has_target"] = has_target
        rec[f"next{w}_lex_score"] = lex_score
        rec[f"next{w}_lex_label"] = lex_label
        rec[f"next{w}_would_change_neutral"] = would_change_neutral
        rec[f"next{w}_would_flip_nonneutral"] = would_flip_nonneutral

    results.append(rec)

result_df = pd.DataFrame(results)

# ------------------------------------------------------------
# COMPUTE SUMMARY STATISTICS
# ------------------------------------------------------------

n_total       = len(result_df)
n_neutral     = (result_df["pipeline_label"] == "neutral").sum()
n_nonneutral  = (result_df["pipeline_label"] != "neutral").sum()
n_article_matched = result_df["article_found"].sum()
n_idx_found   = (result_df["mention_sent_idx"] >= 0).sum()

lines = []
lines.append("=" * 60)
lines.append("SENTENCE WINDOW VALIDATION RESULTS")
lines.append("=" * 60)
lines.append(f"Total pipeline rows (target-mention pairs): {n_total:,}")
lines.append(f"  Neutral:      {n_neutral:,}  ({100*n_neutral/n_total:.1f}%)")
lines.append(f"  Non-neutral:  {n_nonneutral:,}  ({100*n_nonneutral/n_total:.1f}%)")
lines.append(f"Article text found: {n_article_matched:,} / {n_total:,}")
lines.append(f"Mention sentence located in article: {n_idx_found:,} / {n_article_matched:,}")
lines.append("")

for w in NEXT_SENT_WINDOWS:
    lines.append(f"--- Window: +{w} sentence(s) after mention ---")

    # Among neutrals: what fraction have evaluative content in next w sentences?
    neutral_mask = result_df["pipeline_label"] == "neutral"
    has_next = result_df[f"next{w}_text"].notna()

    neutral_checkable = (neutral_mask & has_next).sum()
    neutral_change = result_df[f"next{w}_would_change_neutral"].sum()

    lines.append(f"  DATA LOSS ESTIMATE (neutral rows that have eval in next +{w} sent):")
    lines.append(f"    Neutral rows with retrievable next context: {neutral_checkable:,}")
    lines.append(f"    Of those, next sent has target + eval signal: {neutral_change:,}  "
                 f"({100*neutral_change/max(neutral_checkable,1):.1f}%)")
    lines.append(f"    As % of ALL neutral rows: {100*neutral_change/max(n_neutral,1):.1f}%")

    # Breakdown by domestic/foreign
    for fg_val, fg_label in [(True, "Foreign"), (False, "Domestic")]:
        mask = neutral_mask & has_next & (result_df["is_foreign_final"] == fg_val)
        change = (mask & result_df[f"next{w}_would_change_neutral"]).sum()
        denom = mask.sum()
        lines.append(f"      {fg_label}: {change:,} / {denom:,}  ({100*change/max(denom,1):.1f}%)")

    lines.append("")

    # Among non-neutrals: what fraction would flip direction?
    nonneutral_mask = result_df["pipeline_label"] != "neutral"
    nonneutral_checkable = (nonneutral_mask & has_next).sum()
    nonneutral_flip = result_df[f"next{w}_would_flip_nonneutral"].sum()

    lines.append(f"  NOISE ESTIMATE (non-neutral rows that FLIP with next +{w} sent):")
    lines.append(f"    Non-neutral rows with retrievable next context: {nonneutral_checkable:,}")
    lines.append(f"    Of those, direction flips: {nonneutral_flip:,}  "
                 f"({100*nonneutral_flip/max(nonneutral_checkable,1):.1f}%)")

    for fg_val, fg_label in [(True, "Foreign"), (False, "Domestic")]:
        mask = nonneutral_mask & has_next & (result_df["is_foreign_final"] == fg_val)
        flip = (mask & result_df[f"next{w}_would_flip_nonneutral"]).sum()
        denom = mask.sum()
        lines.append(f"      {fg_label}: {flip:,} / {denom:,}  ({100*flip/max(denom,1):.1f}%)")

    lines.append("")

summary_text = "\n".join(lines)
print("\n" + summary_text)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

Path(OUT_ROW_PATH).parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUT_ROW_PATH, index=False)
Path(OUT_SUM_PATH).write_text(summary_text, encoding="utf-8")

print(f"\nRow-level results: {OUT_ROW_PATH}")
print(f"Summary text:      {OUT_SUM_PATH}")
