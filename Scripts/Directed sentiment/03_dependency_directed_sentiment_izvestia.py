# ============================================================
# 03_dependency_directed_sentiment_izvestia.py
# Directed sentiment in expert mention sentences — Izvestia dataset
#
# Generalization check: same pipeline as 03_dependency_directed_sentiment.py
# applied to the Izvestia Ukraine corpus (Feb 2023 – Mar 2024).
#
# Hybrid method:
#   1. spaCy dependency parsing for target linking
#   2. target-centered / predicate-centered clause extraction
#   3. evaluative lexicon + comparative + blame attribution rules
#   4. transformer sentiment model
#   5. confidence-aware label combination
#
# Input:
#   Data/iz_mentions_with_country.csv
#
# Output:
#   Data/iz_directed_sentiment_full.csv
#   Data/iz_directed_sentiment_long.csv
# ============================================================

import re
import json
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

INPUT_PATH = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/iz_mentions_with_country.csv"

OUTPUT_FULL_PATH = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/iz_directed_sentiment_full.csv"
OUTPUT_LONG_PATH = r"C:/Users/olemga/OneDrive - PRIO/Documents/MA Thesis/Methods/Data/iz_directed_sentiment_long.csv"

SPACY_MODEL = "en_core_web_trf"
# If too slow, use:
# SPACY_MODEL = "en_core_web_md"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

BATCH_SIZE_SPACY = 64
BATCH_SIZE_SENT = 32
MAX_TEXT_LEN = 256

# For testing:
TEST_SAMPLE_N = None
# TEST_SAMPLE_N = 1000

TARGET_WINDOW = 8
PREDICATE_WINDOW = 12

MODEL_STRONG_CONF = 0.72
MODEL_WEAK_CONF = 0.56

# ------------------------------------------------------------
# TARGET DEFINITIONS
# ------------------------------------------------------------

TARGET_PATTERNS = {
    "Russia": [
        "russia", "russian federation", "kremlin", "moscow", "putin"
    ],
    "Ukraine": [
        "ukraine", "ukrainian", "kyiv", "kiev", "zelensky", "zelenskyy"
    ],
    "United States": [
        "united states", "u.s.", "us", "usa", "america", "american", "washington", "white house"
    ],
    "EU": [
        "european union", "eu", "brussels", "european commission", "european parliament"
    ],
    "NATO": [
        "nato", "north atlantic alliance", "alliance"
    ],
    "United Kingdom": [
        "united kingdom", "uk", "britain", "british", "london", "england"
    ],
    "Europe": [
        "europe", "european", "europeans"
    ],
    "West": [
        "the west", "west", "western", "collective west", "western countries", "western allies"
    ]
}

BLOC_MAP = {
    "Russia": "Russia",
    "Ukraine": "Ukraine",
    "United States": "West",
    "EU": "West",
    "NATO": "West",
    "United Kingdom": "West",
    "Europe": "West",
    "West": "West"
}

# ------------------------------------------------------------
# LEXICAL RESOURCES
# ------------------------------------------------------------

SPEECH_VERBS = {
    "say", "said", "state", "stated", "note", "noted", "argue", "argued",
    "claim", "claimed", "stress", "stressed", "add", "added", "explain",
    "explained", "warn", "warned", "announce", "announced", "write", "wrote",
    "tell", "told", "report", "reported", "conclude", "concluded",
    "comment", "commented", "estimate", "estimated", "predict", "predicted",
    "believe", "believed", "think", "thought", "point", "pointed",
    "assess", "assessed", "recall", "recalled", "speak", "spoke"
}

PERSON_ROLE_HEADS = {
    "expert", "analyst", "specialist", "historian", "economist", "politician",
    "scientist", "researcher", "journalist", "professor", "commentator",
    "observer", "official", "minister", "president", "spokesman", "spokesperson",
    "advisor", "adviser", "consultant", "writer", "scholar", "doctor", "coach",
    "diplomat", "ambassador", "senator", "lawyer"
}

LOCATION_PREPS = {"in", "at", "from", "near", "inside", "outside", "around"}
APPOSITION_CUES = {"professor", "expert", "analyst", "consultant", "ambassador", "diplomat", "scientist"}

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
    "collapse", "degrade", "degraded", "threatening", "isolation"
}

NEGATIONS = {"not", "no", "never", "none", "neither", "hardly", "scarcely", "without"}
COMPARATIVE_CUES = {"better", "worse", "stronger", "weaker", "more", "less", "than", "compared", "versus", "vs"}

BLAME_CUES = {
    "because", "because of", "due to", "caused by", "result of",
    "responsible for", "to blame", "led to", "leading to", "provoked by",
    "triggered by", "as a result of"
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def sentiment_to_numeric(label: str):
    return {"negative": -1, "neutral": 0, "positive": 1}.get(label, None)


def tokenize_simple(text: str):
    return re.findall(r"\b[\w'-]+\b", str(text).lower())

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading data...")
df = pd.read_csv(INPUT_PATH)

required_cols = ["expert_id", "row_id", "sentence", "country_final", "is_foreign_final"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[df["sentence"].notna()].copy()
df["sentence"] = df["sentence"].astype(str).map(normalize_spaces)
df = df[df["sentence"].str.len() > 0].reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")

if TEST_SAMPLE_N is not None:
    df = df.sample(TEST_SAMPLE_N, random_state=42).reset_index(drop=True)
    print(f"Testing on sample of {len(df):,} rows")

# ------------------------------------------------------------
# COMPILE TARGET PATTERNS
# ------------------------------------------------------------

compiled_target_patterns = []
for label, phrases in TARGET_PATTERNS.items():
    for phrase in phrases:
        compiled_target_patterns.append(
            (label, phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
        )

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

print(f"Loading spaCy model: {SPACY_MODEL}")
nlp = spacy.load(SPACY_MODEL)

print(f"Loading sentiment model: {SENTIMENT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sent_model.to(device)
sent_model.eval()

sent_labels = ["negative", "neutral", "positive"]

# ------------------------------------------------------------
# SENTIMENT MODEL
# ------------------------------------------------------------

def predict_sentiment(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sent_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    out = []
    for i, pred in enumerate(preds):
        out.append({
            "sentiment_label_model": sent_labels[pred],
            "model_negative": float(probs[i][0]),
            "model_neutral": float(probs[i][1]),
            "model_positive": float(probs[i][2]),
            "model_confidence": float(probs[i][pred])
        })
    return out

# ------------------------------------------------------------
# TARGET FILTERS
# ------------------------------------------------------------

def is_nationality_modifier_of_person(token):
    if token is None:
        return False

    head = token.head
    if head is None or head == token:
        return False

    if token.dep_ in {"amod", "compound"}:
        if head.ent_type_ == "PERSON":
            return True
        if head.lemma_.lower() in PERSON_ROLE_HEADS:
            return True

    return False


def is_likely_location_reference(span):
    """
    Filters uses like:
      'professor at Birkbeck University in London'
      'meeting in Moscow'
    unless there is stronger evaluative structure nearby.
    """
    if span is None or len(span) == 0:
        return False

    root = span.root

    # prepositional object of location phrases
    if root.dep_ == "pobj" and root.head is not None:
        prep = root.head
        if prep.lemma_.lower() in LOCATION_PREPS:
            head = prep.head
            if head is not None and (head.lemma_.lower() in APPOSITION_CUES or head.ent_type_ in {"ORG", "PERSON"}):
                return True

    # appositional/organization biography fragments
    local = span.sent.text.lower()
    if re.search(r"\b(professor|analyst|expert|consultant|ambassador|diplomat)\b.*\b(in|at)\b.*\b" + re.escape(span.text.lower()) + r"\b", local):
        return True

    return False

# ------------------------------------------------------------
# TARGET DETECTION
# ------------------------------------------------------------

def find_target_mentions(doc):
    matches = []

    for label, phrase, pattern in compiled_target_patterns:
        for m in pattern.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if span is None or len(span) == 0:
                continue

            root = span.root

            if is_nationality_modifier_of_person(root):
                continue

            if is_likely_location_reference(span):
                continue

            matches.append({
                "target_fine": label,
                "target_match_text": span.text,
                "target_rule": phrase,
                "target_start": m.start(),
                "target_end": m.end(),
                "span": span,
                "root": root
            })

    deduped = []
    seen = set()
    for m in sorted(matches, key=lambda x: (x["target_start"], x["target_end"], x["target_fine"])):
        key = (m["target_start"], m["target_end"], m["target_fine"])
        if key not in seen:
            deduped.append(m)
            seen.add(key)

    return deduped

# ------------------------------------------------------------
# EVALUATIVE LOGIC
# ------------------------------------------------------------

def lexical_eval_score(text):
    toks = tokenize_simple(text)
    if not toks:
        return 0

    score = 0
    for i, tok in enumerate(toks):
        prev = toks[max(0, i - 3): i]
        negated = any(t in NEGATIONS for t in prev)

        if tok in POSITIVE_EVAL_WORDS:
            score += -1 if negated else 1
        elif tok in NEGATIVE_EVAL_WORDS:
            score += 1 if negated else -1

    return score


def has_comparative_cue(text):
    toks = set(tokenize_simple(text))
    return any(cue in toks for cue in COMPARATIVE_CUES)


def comparative_direction_score(clause_text, target_match_text):
    text = clause_text.lower()
    target = target_match_text.lower()
    idx = text.find(target)
    if idx == -1:
        return 0

    start = max(0, idx - 70)
    end = min(len(text), idx + len(target) + 70)
    local = text[start:end]

    pos_hits = sum(1 for w in POSITIVE_EVAL_WORDS if w in local)
    neg_hits = sum(1 for w in NEGATIVE_EVAL_WORDS if w in local)

    if pos_hits > neg_hits:
        return 1
    if neg_hits > pos_hits:
        return -1
    return 0


def has_blame_cue(text):
    text_l = text.lower()
    return any(cue in text_l for cue in BLAME_CUES)


def blame_direction_score(clause_text, target_match_text):
    """
    If target appears near blame language, treat as negative target framing.
    """
    text = clause_text.lower()
    target = target_match_text.lower()
    idx = text.find(target)
    if idx == -1:
        return 0

    start = max(0, idx - 90)
    end = min(len(text), idx + len(target) + 90)
    local = text[start:end]

    if any(cue in local for cue in BLAME_CUES):
        return -1
    return 0


def clause_is_biographical_or_reporting_only(text):
    """
    Avoid scoring fragments like:
      'according to an expert from the United States'
      'professor at university in London'
      'said the analyst'
    """
    t = text.lower()

    # pure biography / provenance
    if re.search(r"\b(according to|expert from|analyst from|professor at|professor from|consultant in|ambassador to|ambassador in)\b", t):
        if lexical_eval_score(t) == 0 and not has_blame_cue(t):
            return True

    # reporting-only language with no evaluative signal
    toks = set(tokenize_simple(t))
    if toks and toks.issubset(
        {
            "said", "say", "noted", "note", "explained", "explain", "stated", "state",
            "claimed", "claim", "reported", "report", "told", "according", "to",
            "the", "an", "a", "analyst", "expert", "specialist", "that", "from",
            "in", "at", "of", "and", "commented", "predicted", "assessed"
        }
    ):
        return True

    return False

# ------------------------------------------------------------
# DEPENDENCY / CLAUSE LOGIC
# ------------------------------------------------------------

def choose_predicate_for_target(target_token):
    if target_token is None:
        return None

    candidate = None

    if target_token.head is not target_token and target_token.head.pos_ in {"VERB", "AUX", "ADJ", "NOUN"}:
        candidate = target_token.head
    else:
        for anc in target_token.ancestors:
            if anc.pos_ in {"VERB", "AUX", "ADJ", "NOUN"}:
                candidate = anc
                break

    if candidate is None:
        return target_token.sent.root

    # Move away from generic reporting verbs if possible
    if candidate.lemma_.lower() in SPEECH_VERBS:
        for child in candidate.children:
            if child.dep_ in {"ccomp", "xcomp", "advcl", "acl", "relcl"}:
                return child
        for child in candidate.children:
            if child.pos_ in {"VERB", "ADJ", "NOUN"} and child.dep_ not in {"nsubj", "csubj", "punct"}:
                return child

    return candidate


def get_target_window_text(doc, token, window=TARGET_WINDOW):
    start = max(0, token.i - window)
    end = min(len(doc), token.i + window + 1)
    return normalize_spaces(doc[start:end].text)


def get_predicate_window_text(doc, predicate, token, window=PREDICATE_WINDOW):
    if predicate is None:
        return get_target_window_text(doc, token, window=TARGET_WINDOW)

    start = max(0, min(predicate.i, token.i) - window)
    end = min(len(doc), max(predicate.i, token.i) + window + 1)
    return normalize_spaces(doc[start:end].text)


def get_subtree_text(predicate, token):
    if predicate is None:
        return normalize_spaces(token.sent.text)

    subtree_tokens = list(predicate.subtree)
    if not subtree_tokens:
        return normalize_spaces(token.sent.text)

    start = min(t.i for t in subtree_tokens)
    end = max(t.i for t in subtree_tokens)
    return normalize_spaces(predicate.doc[start:end + 1].text)


def choose_best_clause(doc, target_token, predicate):
    target_window = get_target_window_text(doc, target_token, window=TARGET_WINDOW)
    predicate_window = get_predicate_window_text(doc, predicate, target_token, window=PREDICATE_WINDOW)
    subtree_text = get_subtree_text(predicate, target_token)

    candidates = [
        ("predicate_window", predicate_window),
        ("target_window", target_window),
        ("subtree_fallback", subtree_text)
    ]

    scored = []
    for method, text in candidates:
        lex_score = lexical_eval_score(text)
        blame = has_blame_cue(text)
        comp = has_comparative_cue(text)
        biography = clause_is_biographical_or_reporting_only(text)

        score = 0
        if lex_score != 0:
            score += 3
        if blame:
            score += 3
        if comp:
            score += 2
        if biography:
            score -= 4

        # prefer shorter clauses when equally informative
        score -= max(0, len(text.split()) - 18) * 0.05

        scored.append((score, method, text))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_method, best_text = scored[0]

    if best_score < 0:
        return target_window, "target_window_forced"

    return best_text, best_method

# ------------------------------------------------------------
# LABEL COMBINATION
# ------------------------------------------------------------

def combine_sentiment(model_label, model_conf, lex_score, comp_score, blame_score, biography_flag):
    if biography_flag and lex_score == 0 and comp_score == 0 and blame_score == 0:
        return "neutral", "biography_neutral"

    signal = 0
    if lex_score > 0:
        signal += 1
    elif lex_score < 0:
        signal -= 1

    if comp_score > 0:
        signal += 1
    elif comp_score < 0:
        signal -= 1

    if blame_score < 0:
        signal -= 2

    if signal > 0:
        signal_label = "positive"
    elif signal < 0:
        signal_label = "negative"
    else:
        signal_label = "neutral"

    if model_conf >= MODEL_STRONG_CONF:
        if signal_label != "neutral" and signal_label != model_label and abs(signal) >= 2:
            return signal_label, "strong_model_overridden_by_strong_rules"
        return model_label, "model"

    if model_label == "neutral" and signal_label != "neutral":
        return signal_label, "lex_override"

    if model_label in {"positive", "negative"} and model_conf >= MODEL_WEAK_CONF:
        if signal_label != "neutral" and signal_label != model_label and abs(signal) >= 1:
            return signal_label, "lex_contradiction"
        return model_label, "model"

    if signal_label != "neutral":
        return signal_label, "lex_support"

    return "neutral", "neutral_default"

# ------------------------------------------------------------
# DOCUMENT PROCESSING
# ------------------------------------------------------------

def process_doc(doc, original_row):
    rows = []
    mentions = find_target_mentions(doc)

    if not mentions:
        return rows

    used = set()

    for m in mentions:
        target_token = m["root"]
        predicate = choose_predicate_for_target(target_token)
        clause_text, clause_method = choose_best_clause(doc, target_token, predicate)

        key = (
            original_row["row_id"],
            original_row["expert_id"],
            m["target_fine"],
            m["target_start"],
            m["target_end"],
            clause_text
        )
        if key in used:
            continue
        used.add(key)

        row_out = original_row.copy()
        row_out.update({
            "target_fine": m["target_fine"],
            "target_bloc": BLOC_MAP.get(m["target_fine"], "Other"),
            "target_match_text": m["target_match_text"],
            "target_rule": m["target_rule"],
            "target_char_start": m["target_start"],
            "target_char_end": m["target_end"],
            "target_token_text": target_token.text if target_token is not None else None,
            "target_token_dep": target_token.dep_ if target_token is not None else None,
            "target_token_head": target_token.head.text if target_token is not None and target_token.head is not None else None,
            "predicate_text": predicate.text if predicate is not None else None,
            "predicate_lemma": predicate.lemma_ if predicate is not None else None,
            "predicate_pos": predicate.pos_ if predicate is not None else None,
            "linked_clause_text": clause_text,
            "clause_method": clause_method
        })
        rows.append(row_out)

    return rows

# ------------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------------

print("Parsing sentences and extracting targets...")
records = []
rows_as_dicts = df.to_dict("records")

for doc, row in tqdm(
    zip(nlp.pipe(df["sentence"].tolist(), batch_size=BATCH_SIZE_SPACY), rows_as_dicts),
    total=len(df)
):
    records.extend(process_doc(doc, row))

long_df = pd.DataFrame(records)

if long_df.empty:
    print("No directed targets found.")
    full_df = df.copy()
    full_df["has_directed_target"] = 0
    full_df["directed_targets_count"] = 0
    full_df["targets_fine_detected"] = "[]"
    full_df["targets_bloc_detected"] = "[]"
    full_df["directed_targets_json"] = "[]"
    full_df.to_csv(OUTPUT_FULL_PATH, index=False)
    pd.DataFrame().to_csv(OUTPUT_LONG_PATH, index=False)
    raise SystemExit("Finished with no detected targets.")

print(f"Directed target rows: {len(long_df):,}")

# ------------------------------------------------------------
# RULE-BASED SCORES
# ------------------------------------------------------------

print("Scoring lexical, comparative, and blame signals...")

long_df["lex_eval_score"] = long_df["linked_clause_text"].fillna("").map(lexical_eval_score)
long_df["has_eval_words"] = (long_df["lex_eval_score"] != 0).astype(int)
long_df["has_comparative"] = long_df["linked_clause_text"].fillna("").map(has_comparative_cue).astype(int)
long_df["has_blame_cue"] = long_df["linked_clause_text"].fillna("").map(has_blame_cue).astype(int)
long_df["biography_flag"] = long_df["linked_clause_text"].fillna("").map(clause_is_biographical_or_reporting_only).astype(int)

long_df["comparative_score"] = long_df.apply(
    lambda r: comparative_direction_score(r["linked_clause_text"], r["target_match_text"]),
    axis=1
)

long_df["blame_score"] = long_df.apply(
    lambda r: blame_direction_score(r["linked_clause_text"], r["target_match_text"]),
    axis=1
)

# ------------------------------------------------------------
# MODEL SENTIMENT
# ------------------------------------------------------------

print("Running transformer sentiment on linked clauses...")
sent_results = []
texts = long_df["linked_clause_text"].fillna("").astype(str).tolist()

for i in tqdm(range(0, len(texts), BATCH_SIZE_SENT)):
    batch = texts[i:i + BATCH_SIZE_SENT]
    sent_results.extend(predict_sentiment(batch))

sent_df = pd.DataFrame(sent_results)
long_df = pd.concat([long_df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)

# ------------------------------------------------------------
# FINAL LABELS
# ------------------------------------------------------------

print("Combining model and rule-based signals...")

combined = long_df.apply(
    lambda r: combine_sentiment(
        r["sentiment_label_model"],
        r["model_confidence"],
        r["lex_eval_score"],
        r["comparative_score"],
        r["blame_score"],
        bool(r["biography_flag"])
    ),
    axis=1
)

long_df["sentiment_label"] = [x[0] for x in combined]
long_df["sentiment_reason"] = [x[1] for x in combined]
long_df["sentiment_score"] = long_df["sentiment_label"].map(sentiment_to_numeric)

# ------------------------------------------------------------
# PROPAGANDA-FACING VARIABLES
# ------------------------------------------------------------

long_df["pro_russia"] = ((long_df["target_bloc"] == "Russia") & (long_df["sentiment_label"] == "positive")).astype(int)
long_df["anti_russia"] = ((long_df["target_bloc"] == "Russia") & (long_df["sentiment_label"] == "negative")).astype(int)

long_df["pro_west"] = ((long_df["target_bloc"] == "West") & (long_df["sentiment_label"] == "positive")).astype(int)
long_df["anti_west"] = ((long_df["target_bloc"] == "West") & (long_df["sentiment_label"] == "negative")).astype(int)

long_df["pro_ukraine"] = ((long_df["target_bloc"] == "Ukraine") & (long_df["sentiment_label"] == "positive")).astype(int)
long_df["anti_ukraine"] = ((long_df["target_bloc"] == "Ukraine") & (long_df["sentiment_label"] == "negative")).astype(int)

long_df["comparative_pro_russia"] = (
    ((long_df["target_bloc"] == "Russia") & (long_df["sentiment_label"] == "positive")) |
    ((long_df["target_bloc"] == "West") & (long_df["sentiment_label"] == "negative")) |
    ((long_df["target_bloc"] == "Ukraine") & (long_df["sentiment_label"] == "negative"))
).astype(int)

long_df["comparative_pro_west"] = (
    ((long_df["target_bloc"] == "West") & (long_df["sentiment_label"] == "positive")) |
    ((long_df["target_bloc"] == "Russia") & (long_df["sentiment_label"] == "negative"))
).astype(int)

# ------------------------------------------------------------
# ROW-PRESERVING SUMMARY
# ------------------------------------------------------------

print("Building row-preserving summary output...")

summary_records = []

for (expert_id, row_id), group in long_df.groupby(["expert_id", "row_id"], dropna=False):
    summary_records.append({
        "expert_id": expert_id,
        "row_id": row_id,
        "has_directed_target": 1,
        "directed_targets_count": int(len(group)),
        "targets_fine_detected": safe_json(sorted(group["target_fine"].dropna().unique().tolist())),
        "targets_bloc_detected": safe_json(sorted(group["target_bloc"].dropna().unique().tolist())),
        "directed_targets_json": safe_json(group[[
            "target_fine",
            "target_bloc",
            "target_match_text",
            "linked_clause_text",
            "clause_method",
            "predicate_text",
            "sentiment_label_model",
            "model_confidence",
            "lex_eval_score",
            "comparative_score",
            "blame_score",
            "biography_flag",
            "sentiment_label",
            "sentiment_reason",
            "sentiment_score",
            "pro_russia",
            "anti_russia",
            "pro_west",
            "anti_west",
            "pro_ukraine",
            "anti_ukraine",
            "comparative_pro_russia",
            "comparative_pro_west"
        ]].to_dict("records")),

        "any_positive_russia": int(((group["target_bloc"] == "Russia") & (group["sentiment_label"] == "positive")).any()),
        "any_negative_russia": int(((group["target_bloc"] == "Russia") & (group["sentiment_label"] == "negative")).any()),
        "any_neutral_russia": int(((group["target_bloc"] == "Russia") & (group["sentiment_label"] == "neutral")).any()),

        "any_positive_west": int(((group["target_bloc"] == "West") & (group["sentiment_label"] == "positive")).any()),
        "any_negative_west": int(((group["target_bloc"] == "West") & (group["sentiment_label"] == "negative")).any()),
        "any_neutral_west": int(((group["target_bloc"] == "West") & (group["sentiment_label"] == "neutral")).any()),

        "any_positive_ukraine": int(((group["target_bloc"] == "Ukraine") & (group["sentiment_label"] == "positive")).any()),
        "any_negative_ukraine": int(((group["target_bloc"] == "Ukraine") & (group["sentiment_label"] == "negative")).any()),
        "any_neutral_ukraine": int(((group["target_bloc"] == "Ukraine") & (group["sentiment_label"] == "neutral")).any()),

        "any_pro_russia": int((group["pro_russia"] == 1).any()),
        "any_anti_russia": int((group["anti_russia"] == 1).any()),
        "any_pro_west": int((group["pro_west"] == 1).any()),
        "any_anti_west": int((group["anti_west"] == 1).any()),
        "any_pro_ukraine": int((group["pro_ukraine"] == 1).any()),
        "any_anti_ukraine": int((group["anti_ukraine"] == 1).any()),
        "any_comparative_pro_russia": int((group["comparative_pro_russia"] == 1).any()),
        "any_comparative_pro_west": int((group["comparative_pro_west"] == 1).any())
    })

summary_df = pd.DataFrame(summary_records)
full_df = df.merge(summary_df, on=["expert_id", "row_id"], how="left")

zero_fill_cols = [
    "has_directed_target", "directed_targets_count",
    "any_positive_russia", "any_negative_russia", "any_neutral_russia",
    "any_positive_west", "any_negative_west", "any_neutral_west",
    "any_positive_ukraine", "any_negative_ukraine", "any_neutral_ukraine",
    "any_pro_russia", "any_anti_russia",
    "any_pro_west", "any_anti_west",
    "any_pro_ukraine", "any_anti_ukraine",
    "any_comparative_pro_russia", "any_comparative_pro_west"
]

for col in zero_fill_cols:
    if col in full_df.columns:
        full_df[col] = full_df[col].fillna(0).astype(int)

json_fill_cols = ["targets_fine_detected", "targets_bloc_detected", "directed_targets_json"]
for col in json_fill_cols:
    if col in full_df.columns:
        full_df[col] = full_df[col].fillna("[]")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print("Saving outputs...")
Path(OUTPUT_FULL_PATH).parent.mkdir(parents=True, exist_ok=True)

full_df.to_csv(OUTPUT_FULL_PATH, index=False)
long_df.to_csv(OUTPUT_LONG_PATH, index=False)

print("Finished.")
print(f"Saved full file: {OUTPUT_FULL_PATH}")
print(f"Saved long file: {OUTPUT_LONG_PATH}")

# ------------------------------------------------------------
# DIAGNOSTICS
# ------------------------------------------------------------

print("\n--- Sentiment distribution ---")
print(long_df["sentiment_label"].value_counts())

print("\n--- Sentiment by target ---")
print(long_df.groupby(["target_bloc", "sentiment_label"]).size())

print("\n--- Sentiment reason breakdown ---")
print(long_df["sentiment_reason"].value_counts())

print("\n--- Mean sentiment by target ---")
print(long_df.groupby("target_bloc")["sentiment_score"].mean())

print("\n--- Target distribution ---")
print(long_df["target_bloc"].value_counts())

print("\n--- Targets per sentence ---")
print(long_df.groupby(["expert_id", "row_id"]).size().value_counts())

print("\n--- Model confidence summary ---")
print(long_df["model_confidence"].describe())

print("\n--- Negative rate by target ---")
print(long_df.groupby("target_bloc")["sentiment_score"].apply(lambda x: (x == -1).mean()))

print("\n--- Negative sentiment by expert nationality ---")
print(long_df.groupby(["is_foreign_final", "target_bloc"])["sentiment_score"].apply(lambda x: (x == -1).mean()))

