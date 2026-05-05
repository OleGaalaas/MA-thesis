"""
Expert extraction on Izvestia dataset (generalization check)

Adapted from expert_exctraction_1999_2024.py for the Izvestia Ukraine dataset.

Key differences from the Lenta script:
- Text column: 'Content' (instead of news_en)
- Date column: 'DateParsed' (already present, but formatted as "23 March 2023")
- No 'key_role' column — uses a predefined role lexicon instead

Input:  Data/Input/IZ_hele.csv
Output: Data/Output/iz_experts_trf_mentions.csv
        Data/Output/iz_experts_trf_experts.csv
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re
import json
import hashlib
from math import exp

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

# =========================
# Paths & dataset settings
# =========================
CSV_PATH = Path(
    r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input"
) / "IZ_hele.csv"

OUT_PATH = Path(
    r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Output"
) / "iz_experts_trf_mentions.csv"

OUT_EXPERTS_PATH = OUT_PATH.with_name("iz_experts_trf_experts.csv")

TEXT_COL = "Content"
DATE_COL = "DateParsed"

NER_MODEL = "en_core_web_trf"

HOME_COUNTRY = "Russia"

EXTRACTION_VERSION = "v2.0.0-iz"

# =========================
# Predefined role lexicon
# (replaces key_role column from Lenta dataset)
# =========================
PREDEFINED_ROLES = [
    # Generic expert/analyst roles
    "expert", "analyst", "specialist", "consultant", "adviser", "advisor",
    "commentator", "observer", "pundit",
    # Academic roles
    "professor", "researcher", "scientist", "scholar", "academic",
    "lecturer", "doctor", "associate professor",
    # Domain-specific experts
    "political scientist", "political analyst", "political commentator",
    "political expert", "military expert", "military analyst",
    "security expert", "security analyst", "defense expert", "defense analyst",
    "economic expert", "economic analyst", "economist",
    "legal expert", "lawyer", "attorney", "jurist",
    "foreign policy expert", "foreign policy analyst",
    "international relations expert", "geopolitical analyst",
    "energy expert", "energy analyst",
    "financial analyst", "finance expert", "banker",
    "historian", "sociologist", "demographer", "psychologist",
    "public health expert", "epidemiologist", "virologist",
    "cybersecurity expert", "technology expert",
    # Institutional roles (kept to catch government/think-tank affiliates)
    "director", "head", "chief", "deputy director",
    "chairman", "vice chairman", "president", "vice president",
    "minister", "deputy minister", "prime minister",
    "secretary", "secretary general", "deputy secretary",
    "spokesman", "spokesperson",
    "ambassador", "envoy", "commissioner",
    "senator", "representative", "mp", "member of parliament",
    "governor", "mayor",
    "general", "colonel", "lieutenant general", "major general",
    "commander", "admiral",
    "prosecutor", "attorney general",
]


# =========================
# Domain & keyword mapping
# =========================
DOMAIN_CANON = {
    "economist": "economics",
    "economic": "economics",
    "political scientist": "politics",
    "political analyst": "politics",
    "political commentator": "politics",
    "lawyer": "law",
    "jurist": "law",
    "legal expert": "law",
    "public health": "public health",
    "security": "security",
    "defense": "defense",
    "military": "defense",
    "energy": "energy",
    "markets": "markets",
    "finance": "finance",
    "banker": "finance",
    "technology": "technology",
    "cybersecurity": "cybersecurity",
    "data scientist": "data science",
    "russia": "russia studies",
    "europe": "european studies",
    "epidemiology": "epidemiology",
    "virology": "virology",
    "biology": "biology",
    "physics": "physics",
    "chemistry": "chemistry",
    "geography": "geography",
    "climatology": "climatology",
    "sociology": "sociology",
    "sociologist": "sociology",
    "psychology": "psychology",
    "history": "history",
    "historian": "history",
    "demography": "demography",
    "demographer": "demography",
    "statistics": "statistics",
}

SPEECH_VERBS = {
    "say", "tell", "add", "note", "argue", "explain", "write", "report", "claim",
    "state", "announce", "comment", "remark", "declare", "maintain", "assert", "predict",
    "warn", "suggest", "estimate", "reiterate", "underline", "emphasize", "stress"
}
PREV_SENT_CUES = {"according to", "as noted", "as stated", "as reported", "as explained"}

TITLE_TOKENS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "prof.", "professor", "doctor",
    "president", "pm", "prime", "minister", "chancellor", "secretary", "sir",
    "madam", "lord", "lady", "general", "gen", "capt", "captain", "colonel",
    "col", "amb", "ambassador", "governor", "sen", "senator", "rep", "representative",
    "mp", "m.p."
}

PERSON_LABELS = {"PERSON"}

JOURNALIST_KEYWORDS = {
    "correspondent", "reporter", "journalist", "columnist",
    "editor", "host", "anchor"
}

GOV_OFFICIAL_KEYWORDS = {
    "minister", "prime minister", "president", "governor", "mayor",
    "spokesman", "spokesperson", "secretary", "deputy", "chairman",
    "chancellor", "senator", "representative", "mp", "ambassador",
    "envoy", "commissioner", "prosecutor", "attorney general",
    "defense minister", "foreign minister", "interior minister",
    "national security adviser", "security council",
}

ROLE_KEEP_KEYWORDS = {
    "expert", "analyst", "professor", "scientist", "researcher", "specialist",
    "lawyer", "attorney", "minister", "president", "chairman", "director",
    "governor", "mayor", "ambassador", "spokesman", "spokesperson",
    "chief", "head", "officer", "deputy"
}

BAD_PERSON_SURFACES = {
    "vice president",
    "law enforcement",
    "mass media",
    "the expert",
    "the expert opinion",
}

EXPERT_KEYWORDS = {
    "political", "economic", "economics", "security", "military", "strategic",
    "academic", "research", "scholar", "think tank", "university", "institute",
    "professor", "expert", "analyst", "scientist", "specialist"
}
POLICY_KEYWORDS = {
    "policy", "foreign policy", "defense", "sanctions", "war", "conflict",
    "elections", "parliament", "government", "diplomacy", "negotiations"
}


# =========================
# Organization lexicon
# =========================
ORG_TAIL_TERMS = {
    "university", "college", "school", "academy", "institute", "laboratory", "lab",
    "department", "faculty",
    "center", "centre", "programme", "program", "unit",
    "ministry", "council", "committee", "commission", "authority", "agency", "directorate",
    "foundation", "society", "association", "forum", "network", "observatory",
    "institution", "consortium", "panel", "board",
    "bank", "fund",
    "newspaper", "magazine", "journal", "press", "television", "tv", "radio", "news",
    "channel", "telegram", "youtube"
}
SMALL_WORDS = {"of", "for", "in", "on", "at", "and", "the", "a", "an", "to", "with", "from", "under"}
ORG_PREPS = {"at", "of", "in", "for", "from", "with", "under"}

MEDIA_WORDS = {
    "newspaper", "daily", "magazine", "journal", "press", "television", "tv", "radio",
    "news", "agency", "youtube", "telegram", "channel"
}

ACRONYM_MAP = {
    "NSC": "National Security Council",
    "NATO": "NATO",
    "UN": "United Nations",
    "EU": "European Union",
    "FBI": "Federal Bureau of Investigation",
    "CIA": "Central Intelligence Agency",
    "MoD": "Ministry of Defence",
}

def expand_acronym(s: str) -> str:
    t = s.strip()
    return ACRONYM_MAP.get(t, t)

NON_PERSON_HEADS = {
    "university", "institute", "academy", "school", "college",
    "ministry", "government", "council", "committee", "commission",
    "agency", "department", "bank", "fund", "force", "army", "navy",
    "air", "economy", "parliament", "assembly", "union", "federation",
    "asia", "europe", "america", "africa", "middle", "east", "west",
}


# =========================
# Country helpers / lookup layer
# =========================
ORG_COUNTRY_MAP = {
    "Harvard University": "United States",
    "Massachusetts Institute of Technology": "United States",
    "MIT": "United States",
    "Stanford University": "United States",
    "Yale University": "United States",
    "Princeton University": "United States",
    "Columbia University": "United States",
    "University of Chicago": "United States",
    "University of California, Berkeley": "United States",
    "University of California": "United States",
    "New York University": "United States",
    "Georgetown University": "United States",
    "Johns Hopkins University": "United States",
    "Harvard Kennedy School": "United States",
    "Brookings Institution": "United States",
    "Carnegie Endowment for International Peace": "United States",
    "RAND Corporation": "United States",
    "Council on Foreign Relations": "United States",
    "Hoover Institution": "United States",
    "Atlantic Council": "United States",
    "Heritage Foundation": "United States",
    "Nixon Center": "United States",
    "World Bank": "International",
    "International Monetary Fund": "International",
    "IMF": "International",
    "Federal Reserve": "United States",
    "White House": "United States",
    "FBI": "United States",
    "CIA": "United States",
    "CIBC Oppenheimer": "Canada",
    "University of Oxford": "United Kingdom",
    "University of Cambridge": "United Kingdom",
    "Cambridge University": "United Kingdom",
    "London School of Economics": "United Kingdom",
    "LSE": "United Kingdom",
    "King's College London": "United Kingdom",
    "University College London": "United Kingdom",
    "UCL": "United Kingdom",
    "Chatham House": "United Kingdom",
    "Royal United Services Institute": "United Kingdom",
    "RUSI": "United Kingdom",
    "BBC": "United Kingdom",
    "Royal School of Mining": "United Kingdom",
    "University of Reading": "United Kingdom",
    "Sciences Po": "France",
    "National Centre for Scientific Research": "France",
    "CNRS": "France",
    "Sorbonne University": "France",
    "Max Planck": "Germany",
    "Humboldt University of Berlin": "Germany",
    "Free University of Berlin": "Germany",
    "Konrad Adenauer Foundation": "Germany",
    "Friedrich Ebert Foundation": "Germany",
    "Die Zeit": "Germany",
    "Süddeutsche Zeitung": "Germany",
    "European Central Bank": "International",
    "European Commission": "International",
    "European Parliament": "International",
    "European Union": "International",
    "NATO": "International",
    "United Nations": "International",
    "UN Security Council": "International",
    "World Trade Organization": "International",
    "WTO": "International",
    "Organization for Security and Co-operation in Europe": "International",
    "OSCE": "International",
    "Government of Russia": "Russia",
    "Government of the Russian Federation": "Russia",
    "State Duma": "Russia",
    "Federation Council": "Russia",
    "Russian Academy of Sciences": "Russia",
    "Higher School of Economics": "Russia",
    "HSE University": "Russia",
    "MGIMO University": "Russia",
    "RANEPA": "Russia",
    "Skolkovo Institute of Science and Technology": "Russia",
    "Financial University under the Government of Russia": "Russia",
    "Financial University under the Government of the Russian Federation": "Russia",
    "RIA Novosti": "Russia",
    "Interfax": "Russia",
    "TASS": "Russia",
    "ITAR-TASS": "Russia",
    "National Bank of Ukraine": "Ukraine",
    "Peking University": "China",
    "Tsinghua University": "China",
    "Fudan University": "China",
    "Renmin University": "China",
    "Chinese Academy of Social Sciences": "China",
    "Xinhua": "China",
    "University of Tokyo": "Japan",
    "Tokyo University": "Japan",
    "Keio University": "Japan",
    "Waseda University": "Japan",
    "Tel Aviv University": "Israel",
    "Hebrew University of Jerusalem": "Israel",
    "Bar-Ilan University": "Israel",
    "Stockholm International Peace Research Institute": "Sweden",
    "SIPRI": "Sweden",
    "Norwegian Institute of International Affairs": "Norway",
    "NUPI": "Norway",
    "Leiden University": "Netherlands",
    "KU Leuven": "Belgium",
    "University of Geneva": "Switzerland",
    "Graduate Institute Geneva": "Switzerland",
    "Central European University": "Austria",
    "Byrd Polar Research Center": "United States",
    "Reuters": "International",
    "Associated Press": "United States",
    "AP": "United States",
    "USA Today": "United States",
    "BBC News": "United Kingdom",
    # Czech institutions relevant for this dataset
    "Brno Defense University": "Czechia",
    "University of Defense": "Czechia",
}

NORP_COUNTRY_MAP = {
    "american": "United States", "u.s.": "United States", "us": "United States", "usa": "United States",
    "british": "United Kingdom", "english": "United Kingdom", "scottish": "United Kingdom",
    "welsh": "United Kingdom",
    "irish": "Ireland",
    "french": "France",
    "german": "Germany",
    "spanish": "Spain",
    "italian": "Italy",
    "dutch": "Netherlands",
    "norwegian": "Norway",
    "swedish": "Sweden",
    "danish": "Denmark",
    "finnish": "Finland",
    "polish": "Poland",
    "russian": "Russia",
    "ukrainian": "Ukraine",
    "belarusian": "Belarus",
    "georgian": "Georgia",
    "armenian": "Armenia",
    "azerbaijani": "Azerbaijan",
    "kazakh": "Kazakhstan",
    "uzbek": "Uzbekistan",
    "kyrgyz": "Kyrgyzstan",
    "tajik": "Tajikistan",
    "turkmen": "Turkmenistan",
    "moldovan": "Moldova",
    "romanian": "Romania",
    "bulgarian": "Bulgaria",
    "serbian": "Serbia",
    "croatian": "Croatia",
    "slovak": "Slovakia",
    "slovene": "Slovenia",
    "estonian": "Estonia",
    "latvian": "Latvia",
    "lithuanian": "Lithuania",
    "czech": "Czechia",
    "hungarian": "Hungary",
    "chinese": "China",
    "japanese": "Japan",
    "korean": "South Korea", "south korean": "South Korea", "north korean": "North Korea",
    "indian": "India",
    "pakistani": "Pakistan",
    "canadian": "Canada",
    "australian": "Australia",
    "brazilian": "Brazil",
    "mexican": "Mexico",
    "turkish": "Türkiye",
    "iranian": "Iran",
    "israeli": "Israel",
    "egyptian": "Egypt",
    "south african": "South Africa",
    "argentine": "Argentina",
    "kenyan": "Kenya",
    "swiss": "Switzerland",
    "austrian": "Austria",
}

COUNTRY_NAMES = {
    "United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands",
    "Norway", "Sweden", "Denmark", "Finland", "Poland", "Russia", "Ukraine", "Belarus",
    "Georgia", "Armenia", "Azerbaijan", "Kazakhstan", "Uzbekistan", "Kyrgyzstan",
    "Tajikistan", "Turkmenistan", "Moldova", "Romania", "Bulgaria", "Serbia", "Croatia",
    "Slovakia", "Slovenia", "Lithuania", "Latvia", "Estonia", "Czechia", "Hungary",
    "Switzerland", "Austria", "Belgium", "Ireland", "Portugal", "Greece",
    "Turkey", "Türkiye", "Israel", "Egypt", "South Africa", "Kenya",
    "China", "Japan", "South Korea", "North Korea", "India", "Pakistan",
    "Canada", "Australia", "Brazil", "Mexico", "Argentina", "Singapore",
    "Taiwan", "Cyprus", "Malta", "Luxembourg", "Iceland",
    "International",
}

CITY_TO_COUNTRY = {
    "london": "United Kingdom", "oxford": "United Kingdom", "cambridge": "United Kingdom",
    "paris": "France", "berlin": "Germany", "munich": "Germany", "münchen": "Germany",
    "rome": "Italy", "milan": "Italy", "madrid": "Spain", "barcelona": "Spain",
    "oslo": "Norway", "stockholm": "Sweden", "copenhagen": "Denmark", "helsinki": "Finland",
    "warsaw": "Poland", "amsterdam": "Netherlands", "brussels": "Belgium",
    "vienna": "Austria", "zurich": "Switzerland", "geneva": "Switzerland",
    "prague": "Czechia", "budapest": "Hungary",
    "washington": "United States", "washington dc": "United States",
    "new york": "United States", "boston": "United States", "los angeles": "United States",
    "chicago": "United States", "san francisco": "United States",
    "ottawa": "Canada", "toronto": "Canada", "vancouver": "Canada",
    "mexico city": "Mexico",
    "sao paulo": "Brazil", "rio de janeiro": "Brazil",
    "buenos aires": "Argentina",
    "beijing": "China", "peking": "China", "shanghai": "China",
    "tokyo": "Japan", "seoul": "South Korea", "singapore": "Singapore",
    "hong kong": "China", "taipei": "Taiwan",
    "delhi": "India", "mumbai": "India",
    "jerusalem": "Israel", "tel aviv": "Israel",
    "cairo": "Egypt",
    "istanbul": "Türkiye", "ankara": "Türkiye",
    "johannesburg": "South Africa", "nairobi": "Kenya",
    "moscow": "Russia", "saint petersburg": "Russia", "st petersburg": "Russia",
    "kyiv": "Ukraine", "kiev": "Ukraine",
    "brno": "Czechia", "bratislava": "Slovakia",
}

ALIAS_TO_COUNTRY = {
    "u.s.": "United States", "u.s": "United States", "us": "United States", "usa": "United States",
    "u.k.": "United Kingdom", "u.k": "United Kingdom", "uk": "United Kingdom",
}

LOCATION_PREPS = {"in", "to", "into", "at", "towards", "onto"}

ROLE_SURFACES_LOWER: set = set()


# =========================
# Utility helpers
# =========================
def normalize_ws(s: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", " ", s.strip()) if s else s

def normalize_for_id(text: str) -> str:
    if not text:
        return ""
    t = normalize_ws(text).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()

def hash_id(prefix: str, key: str, n: int = 10) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"

def normalize_to_country(name: str) -> str:
    if not name:
        return "Unknown"
    s = normalize_ws(name) or ""
    if s in COUNTRY_NAMES:
        return s
    low = s.lower().strip(" .,")
    if low in ALIAS_TO_COUNTRY:
        return ALIAS_TO_COUNTRY[low]
    if low in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[low]
    for c in COUNTRY_NAMES:
        if re.search(rf"\b{re.escape(c)}\b", s, flags=re.I):
            return c
    for dem, c in NORP_COUNTRY_MAP.items():
        if re.search(rf"\b{re.escape(dem)}\b", s, flags=re.I):
            return c
    return "Unknown"

def strip_leading_titles(text: str) -> str:
    toks = text.strip().split()
    i = 0
    while i < len(toks) and toks[i].lower().rstrip(".") in TITLE_TOKENS:
        i += 1
    toks = toks[i:] if i < len(toks) else toks
    return " ".join(toks) if toks else text

def title_case_org(s: str) -> str:
    parts = s.split()
    out = []
    for i, w in enumerate(parts):
        lw = w.lower()
        if lw in SMALL_WORDS and i != 0:
            out.append(lw)
        else:
            out.append(w[0].upper() + w[1:] if w else w)
    return " ".join(out)


# =========================
# PERSON heuristics
# =========================
def looks_like_person_span(span: "spacy.tokens.Span") -> bool:
    text = span.text.strip()
    if all(t.pos_ == "PRON" for t in span):
        return False
    if getattr(span, "label_", "") and span.label_ and span.label_ != "PERSON":
        return False
    non_person_hits = 0
    for t in span:
        if t.ent_type_ in {"ORG", "GPE", "LOC", "NORP", "FAC"}:
            non_person_hits += 1
    if non_person_hits >= max(2, len(span) // 2 + 1):
        return False
    if len(span) == 1:
        t = span[0]
        if t.pos_ in {"PRON", "DET"}:
            return False
        if not t.text[:1].isupper():
            return False
    if not any(t.pos_ == "PROPN" for t in span):
        return False
    low = text.lower()
    if low in ROLE_SURFACES_LOWER:
        return False
    if text.islower():
        return False
    if span[-1].lemma_.lower() in NON_PERSON_HEADS:
        return False
    if text.isupper() and len(text) > 2:
        return False
    if low in BAD_PERSON_SURFACES:
        return False
    return True


def build_surname_memory(doc: "spacy.tokens.Doc") -> Dict[str, str]:
    memory: Dict[str, Dict[str, int]] = {}
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        txt = strip_leading_titles(ent.text)
        toks = [t for t in txt.split() if t]
        if len(toks) < 2:
            continue
        surname = toks[-1]
        memory.setdefault(surname, {})
        memory[surname][txt] = memory[surname].get(txt, 0) + 1
    best: Dict[str, str] = {}
    for s, cand in memory.items():
        best_full = max(cand.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        best[s] = best_full
    return best

def canonicalize_person_name(name: str, surname_memory: Dict[str, str]) -> str:
    txt = strip_leading_titles(name)
    toks = txt.split()
    if len(toks) == 1:
        s = toks[0]
        if s in surname_memory:
            return surname_memory[s]
    return txt


def find_rule_based_person_spans(
    sent: "spacy.tokens.Span",
    doc: "spacy.tokens.Doc",
    existing_spans: List["spacy.tokens.Span"],
) -> List["spacy.tokens.Span"]:
    used = set()
    for sp in existing_spans:
        used.update(range(sp.start, sp.end))

    spans: List["spacy.tokens.Span"] = []
    i = sent.start
    while i < sent.end:
        tok = doc[i]
        if (
            tok.text.lower().rstrip(".") in TITLE_TOKENS
            and tok.i not in used
            and i + 1 < sent.end
        ):
            j = i + 1
            while (
                j < sent.end
                and j not in used
                and doc[j].text[:1].isupper()
                and doc[j].is_alpha
            ):
                j += 1
            if j > i + 1:
                span = doc[i:j]
                if looks_like_person_span(span):
                    spans.append(span)
                    used.update(range(i, j))
                    i = j
                    continue
        if (
            i not in used
            and tok.text[:1].isupper()
            and tok.is_alpha
            and tok.pos_ in {"PROPN", "NOUN"}
        ):
            j = i + 1
            while (
                j < sent.end
                and j not in used
                and doc[j].text[:1].isupper()
                and doc[j].is_alpha
                and doc[j].pos_ in {"PROPN", "NOUN"}
            ):
                j += 1
            length = j - i
            if 2 <= length <= 3:
                span = doc[i:j]
                if looks_like_person_span(span):
                    spans.append(span)
                    used.update(range(i, j))
            i = j
        else:
            i += 1
    return spans


# =========================
# PhraseMatcher from role list
# =========================
def build_phrase_matcher(nlp, role_list: List[str]) -> PhraseMatcher:
    m = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(s) for s in role_list if s]
    m.add("EXPERT_ROLE", patterns)
    return m


# =========================
# Organization extraction
# =========================
def chunk_has_org_signal(nc: "spacy.tokens.Span") -> bool:
    words = [t.text.lower() for t in nc]
    if any(w in ORG_TAIL_TERMS for w in words):
        return True
    txt = nc.text.lower()
    if re.search(
        r"\b(university|institute|center|centre|college|academy|ministry|fund|bank|agency|council|committee|channel|youtube|telegram)\b",
        txt,
    ):
        return True
    return False

def reconstruct_org_from_chunk(nc: "spacy.tokens.Span") -> str:
    toks = list(nc)
    if len(toks) == 1 and toks[0].text in ACRONYM_MAP:
        return expand_acronym(toks[0].text)
    keep = []
    for t in toks:
        if t.pos_ in {"PROPN", "NOUN", "ADJ"} or t.dep_ in {"compound", "amod"} or t.text.lower() in SMALL_WORDS:
            keep.append(t.text)
        elif t.is_punct and t.text in {"&", "-", "/", "'", "'"}:
            keep.append(t.text)
    candidate = normalize_ws(" ".join(keep)) or nc.text
    candidate = title_case_org(candidate)
    parts = candidate.split()
    while parts and parts[-1].lower() in SMALL_WORDS:
        parts.pop()
    candidate = " ".join(parts) if parts else candidate
    parts = candidate.split()
    parts = [expand_acronym(p) if p in ACRONYM_MAP else p for p in parts]
    candidate = " ".join(parts)
    if not any(t.pos_ == "PROPN" for t in nc):
        return "Unknown"
    return candidate

def organization_from_role_context(
    sent: "spacy.tokens.Span",
    role_span: "spacy.tokens.Span"
) -> Optional[str]:
    doc = sent.doc
    head = role_span.root
    for child in head.children:
        if child.dep_ == "prep" and child.text.lower() in ORG_PREPS:
            pobj = next((c2 for c2 in child.children if c2.dep_ in {"pobj", "obj"}), None)
            if pobj is not None:
                for nc in sent.noun_chunks:
                    if nc.start <= pobj.i < nc.end and chunk_has_org_signal(nc):
                        org = reconstruct_org_from_chunk(nc)
                        if org != "Unknown":
                            return org
    end_i = role_span.end
    if end_i < sent.end and doc[end_i].text.lower() in ORG_PREPS:
        j = end_i + 1
        while j < sent.end and doc[j].text != ",":
            j += 1
        if j > end_i + 1:
            span = doc[end_i:j]
            candidates = []
            for nc in sent.noun_chunks:
                if not (nc.end <= span.start or nc.start >= span.end):
                    if chunk_has_org_signal(nc):
                        candidates.append(nc)
            if candidates:
                best = max(candidates, key=lambda s: s.end - s.start)
                org = reconstruct_org_from_chunk(best)
                if org != "Unknown":
                    return org
    return None

def fallback_sentence_org(
    sent: "spacy.tokens.Span",
    role_span: "spacy.tokens.Span",
    person_span: Optional["spacy.tokens.Span"]
) -> Optional[str]:
    anchor = role_span.start
    if person_span is not None and abs(person_span.start - role_span.start) < 5:
        anchor = person_span.start
    org_like = [nc for nc in sent.noun_chunks if chunk_has_org_signal(nc)]
    if not org_like:
        return None
    after = [nc for nc in org_like if nc.start >= anchor]
    before = [nc for nc in org_like if nc.start < anchor]
    candidates = (after[:3] if after else before[-3:]) or org_like
    best = min(candidates, key=lambda nc: abs(nc.start - anchor))
    org = reconstruct_org_from_chunk(best)
    return org if org != "Unknown" else None


# =========================
# Role→person linking
# =========================
def score_person_for_role(
    sent: "spacy.tokens.Span",
    role_span: "spacy.tokens.Span",
    person_span: "spacy.tokens.Span"
) -> float:
    score = 0.0
    head = role_span.root
    same_nc = any(
        nc.start <= role_span.start < nc.end and nc.start <= person_span.start < nc.end
        for nc in sent.noun_chunks
    )
    if same_nc:
        score += 6.0
    if person_span.root.dep_ == "appos" or head.dep_ == "appos":
        score += 4.0
    for tok in sent:
        if tok.lemma_ in SPEECH_VERBS and tok.pos_ in {"VERB", "AUX"}:
            if any(
                ch.dep_ in {"nsubj", "nsubjpass"} and person_span.start <= ch.i < person_span.end
                for ch in tok.children
            ):
                score += 3.5
                break
    for tok in person_span:
        if tok.dep_ in {"pobj", "obj"} and tok.head.dep_ == "prep" and tok.head.text.lower() in {"about", "on", "against"}:
            score -= 3.0
            break
    dist = min(abs(person_span.start - role_span.start), 20)
    score += 2.5 * exp(-0.30 * dist)
    if person_span.start >= role_span.end:
        gap = person_span.start - role_span.end
        if gap <= 3:
            score += 1.2
    elif role_span.start >= person_span.end:
        gap = role_span.start - person_span.end
        if gap <= 3:
            score += 0.8
    if person_span.start < role_span.start:
        score += 0.3
    return score

def pick_best_person(
    sent: "spacy.tokens.Span",
    role_span: "spacy.tokens.Span",
    person_spans: List["spacy.tokens.Span"]
) -> Optional["spacy.tokens.Span"]:
    best = None
    best_score = float("-inf")
    for p in person_spans:
        sc = score_person_for_role(sent, role_span, p)
        if sc > best_score:
            best, best_score = p, sc
    if best is not None:
        if best.start == role_span.start and best.end == role_span.end:
            return None
    return best if best and best_score >= 2.0 else None

def link_evidence_type(
    sent: "spacy.tokens.Span",
    role_span: "spacy.tokens.Span",
    person_span: Optional["spacy.tokens.Span"],
    link_source: str
) -> Tuple[str, str]:
    if person_span is None:
        return "none", ""
    if link_source == "prev_sent":
        return "prev_sentence", "linked via previous sentence cue/speaker heuristic"
    if person_span.root.dep_ == "appos" or role_span.root.dep_ == "appos":
        return "appos", "appositional construction"
    same_nc = any(
        nc.start <= role_span.start < nc.end and nc.start <= person_span.start < nc.end
        for nc in sent.noun_chunks
    )
    if same_nc:
        return "noun_chunk", "same noun chunk"
    for tok in sent:
        if tok.lemma_ in SPEECH_VERBS and tok.pos_ in {"VERB", "AUX"}:
            if any(
                ch.dep_ in {"nsubj", "nsubjpass"} and person_span.start <= ch.i < person_span.end
                for ch in tok.children
            ):
                return "nsubj_speech", f"subject of speech verb '{tok.text}'"
    return "proximity", "best proximity/heuristic score"


# =========================
# Country cue generation (Detection)
# =========================
def _build_of_from_patterns() -> List[Tuple[str, str]]:
    pats = []
    for c in sorted(COUNTRY_NAMES, key=len, reverse=True):
        if c == "International":
            continue
        pats.append((c, rf"\b(?:of|from)\s+{re.escape(c)}\b"))
        pats.append((c, rf"\bunder\s+(?:the\s+)?government\s+of\s+{re.escape(c)}\b"))
    for dem, c in NORP_COUNTRY_MAP.items():
        pats.append((c, rf"\b(?:of|from)\s+(?:the\s+)?{re.escape(dem)}\b"))
    for alias, c in ALIAS_TO_COUNTRY.items():
        pats.append((c, rf"\b(?:of|from)\s+(?:the\s+)?{re.escape(alias)}\b"))
    pats.append(("Russia", r"\bof\s+(?:the\s+)?russian\s+federation\b"))
    return pats

OF_FROM_PATTERNS = _build_of_from_patterns()

def detect_country_cues(
    org_text: str,
    sent: "spacy.tokens.Span",
    person_span: Optional["spacy.tokens.Span"],
) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    sentence_text = sent.text

    if org_text and org_text != "Unknown":
        low_org = org_text.lower()
        for key, country in ORG_COUNTRY_MAP.items():
            if re.search(rf"\b{re.escape(key.lower())}\b", low_org):
                cues.append({
                    "country": normalize_to_country(country),
                    "cue_type": "org_lookup",
                    "strength": 1.00,
                    "evidence": f"org match: {key}",
                })

    if org_text and org_text != "Unknown":
        for c in COUNTRY_NAMES:
            if c == "International":
                continue
            if re.search(rf"\b{re.escape(c)}\b", org_text, flags=re.I):
                cues.append({
                    "country": normalize_to_country(c),
                    "cue_type": "org_embedded",
                    "strength": 0.90,
                    "evidence": f"in org: {c}",
                })
        for dem, c in NORP_COUNTRY_MAP.items():
            if re.search(rf"\b{re.escape(dem)}\b", org_text, flags=re.I):
                cues.append({
                    "country": normalize_to_country(c),
                    "cue_type": "org_embedded",
                    "strength": 0.90,
                    "evidence": f"in org demonym: {dem}",
                })

    for c, pat in OF_FROM_PATTERNS:
        m = re.search(pat, sentence_text, flags=re.I)
        if m:
            cues.append({
                "country": normalize_to_country(c),
                "cue_type": "syntax_of_from",
                "strength": 0.80,
                "evidence": m.group(0),
            })
            break

    def ents_near_person(labels: set, window_tokens: int = 8):
        if person_span is None:
            return []
        lo = max(sent.start, person_span.start - window_tokens)
        hi = min(sent.end, person_span.end + window_tokens)
        hits = []
        for ent in sent.ents:
            if ent.label_ in labels and not (ent.end <= lo or ent.start >= hi):
                hits.append(ent)
        return hits

    for ent in ents_near_person({"NORP"}):
        dem = ent.text.lower()
        if dem in NORP_COUNTRY_MAP:
            cues.append({
                "country": normalize_to_country(NORP_COUNTRY_MAP[dem]),
                "cue_type": "norp_near_person",
                "strength": 0.75,
                "evidence": f"NORP near person: {ent.text}",
            })

    for ent in sent.ents:
        if ent.label_ == "NORP":
            dem = ent.text.lower()
            if dem in NORP_COUNTRY_MAP:
                cues.append({
                    "country": normalize_to_country(NORP_COUNTRY_MAP[dem]),
                    "cue_type": "norp",
                    "strength": 0.70,
                    "evidence": f"NORP: {ent.text}",
                })

    def is_location_pp(ent):
        if ent.start > sent.start:
            prev = ent.doc[ent.start - 1].text.lower()
            if prev in LOCATION_PREPS:
                return True
        return False

    for ent in ents_near_person({"GPE"}):
        if not is_location_pp(ent):
            norm = normalize_to_country(ent.text)
            strength = 0.60 if norm in COUNTRY_NAMES else 0.55
            cues.append({
                "country": norm,
                "cue_type": "gpe_near_person",
                "strength": strength,
                "evidence": f"GPE near person: {ent.text}",
            })

    for ent in sent.ents:
        if ent.label_ == "GPE" and not is_location_pp(ent):
            norm = normalize_to_country(ent.text)
            strength = 0.55 if norm in COUNTRY_NAMES else 0.50
            cues.append({
                "country": norm,
                "cue_type": "gpe",
                "strength": strength,
                "evidence": f"GPE: {ent.text}",
            })

    cues = [c for c in cues if c.get("country") and c["country"] != "Unknown"]
    return cues


# =========================
# Country decision (Decision)
# =========================
CUE_PRIORITY = {
    "org_lookup": 5,
    "org_embedded": 4,
    "syntax_of_from": 4,
    "norp_near_person": 3,
    "norp": 2,
    "gpe_near_person": 2,
    "gpe": 1,
}

def decide_mention_country(
    cues: List[Dict[str, Any]],
    ambiguity_delta: float = 0.10,
    min_strength: float = 0.55,
) -> Tuple[str, float, str, str, int, float]:
    if not cues:
        return "Unknown", 0.0, "unknown", "", 1, 0.0
    by_country: Dict[str, List[Dict[str, Any]]] = {}
    for cue in cues:
        by_country.setdefault(cue["country"], []).append(cue)
    scored: List[Tuple[str, float, float, List[Dict[str, Any]]]] = []
    for country, items in by_country.items():
        best_item = max(
            items,
            key=lambda c: (CUE_PRIORITY.get(c["cue_type"], 0), float(c["strength"]))
        )
        prio = CUE_PRIORITY.get(best_item["cue_type"], 0)
        strength = float(best_item["strength"])
        score = prio + strength
        scored.append((country, score, strength, items))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_country, best_score, best_strength, best_items = scored[0]
    best_sources = sorted({c["cue_type"] for c in best_items})
    best_evidence = " | ".join([c["evidence"] for c in best_items][:6])
    second_score = scored[1][1] if len(scored) > 1 else None
    top_margin = (best_score - second_score) if second_score is not None else best_score
    ambiguous = 0
    if best_strength < min_strength:
        ambiguous = 1
    elif second_score is not None and top_margin <= ambiguity_delta:
        ambiguous = 1
    cue_types = {c["cue_type"] for c in best_items}
    confidence = min(1.0, best_strength + 0.05 * (len(cue_types) - 1))
    return best_country, confidence, "+".join(best_sources), best_evidence, ambiguous, float(top_margin)


# =========================
# Role classification
# =========================
def classify_org_type(org: str) -> Tuple[str, int]:
    if not org or org == "Unknown":
        return "unknown", 0
    low = org.lower()
    if any(w in low for w in MEDIA_WORDS):
        return "media", 1
    if any(t in low for t in ["university", "institute", "academy", "school", "college", "laboratory", "lab"]):
        return "academic", 0
    if any(t in low for t in ["ministry", "government", "council", "agency", "committee", "department", "directorate", "authority"]):
        return "government", 0
    return "other", 0

def classify_role_class(role_text: str, org_type: str) -> str:
    rl = (role_text or "").lower()
    is_media = any(k in rl for k in JOURNALIST_KEYWORDS) or (org_type == "media")
    if is_media:
        return "media_actor"
    is_gov = any(k in rl for k in GOV_OFFICIAL_KEYWORDS) or (org_type == "government")
    if is_gov:
        return "government_official"
    if any(k in rl for k in ["professor", "researcher", "scientist", "scholar", "academic", "lecturer"]):
        return "academic_expert"
    if org_type == "academic":
        return "academic_expert"
    if any(k in rl for k in ["analyst", "expert", "political", "security", "defense", "strategic", "think tank", "consultant"]):
        return "policy_expert"
    return "other"


# =========================
# Domain inference
# =========================
def infer_domain_from_role(role_text: str) -> Tuple[str, str, float]:
    role_low = (role_text or "").lower()
    domain = "Unknown"
    for k, v in sorted(DOMAIN_CANON.items(), key=lambda kv: -len(kv[0])):
        if k in role_low:
            domain = v
            break
    if domain != "Unknown":
        return domain, "role", 1.0
    return "Unknown", "unknown", 0.0


# =========================
# Extraction per doc
# =========================
def extract_expert_rows_from_doc(
    doc: "spacy.tokens.Doc",
    row_id: int,
    date: str,
    matcher: PhraseMatcher,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sents = list(doc.sents)
    surname_memory = build_surname_memory(doc)

    ents_by_sent: Dict["spacy.tokens.Span", List["spacy.tokens.Span"]] = {s: [] for s in sents}
    for e in doc.ents:
        if e.label_ in PERSON_LABELS and looks_like_person_span(e):
            for s in sents:
                if e.start >= s.start and e.end <= s.end:
                    ents_by_sent[s].append(e)
                    break

    for i, sent in enumerate(sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        matches = matcher(sent)
        if not matches:
            continue

        role_spans: List["spacy.tokens.Span"] = []
        for _, start, end in matches:
            span = doc[start:end]
            for nc in sent.noun_chunks:
                if nc.start <= span.start and nc.end >= span.end:
                    span = doc[nc.start:nc.end]
                    break
            if not any(
                (rs.start <= span.start < rs.end) or (span.start <= rs.start < span.end)
                for rs in role_spans
            ):
                role_spans.append(span)

        if not role_spans:
            continue

        cand_persons = ents_by_sent.get(sent, []).copy()
        rb_persons = find_rule_based_person_spans(sent, doc, cand_persons)
        if rb_persons:
            cand_persons.extend(rb_persons)

        experts_in_sentence_count = len(role_spans)
        sentence_has_multiple_experts = int(experts_in_sentence_count >= 2)

        nc_to_person: Dict["spacy.tokens.Span", "spacy.tokens.Span"] = {}

        for role_span in role_spans:
            nc_for_role = None
            for nc in sent.noun_chunks:
                if nc.start <= role_span.start < nc.end:
                    nc_for_role = nc
                    break

            best_p = pick_best_person(sent, role_span, cand_persons)
            link_source = "intra_sent" if best_p is not None else "none"

            if best_p is None and i > 0:
                lower = sent_text.lower()
                prev_sent = sents[i - 1]
                prev_ps = ents_by_sent.get(prev_sent, []).copy()
                rb_prev_persons = find_rule_based_person_spans(prev_sent, doc, prev_ps)
                if rb_prev_persons:
                    prev_ps.extend(rb_prev_persons)
                has_speech = any(
                    t.lemma_ in SPEECH_VERBS and t.pos_ in {"VERB", "AUX"} for t in prev_sent
                )
                cue_cond = (
                    any(cue in lower for cue in PREV_SENT_CUES)
                    or re.match(r"^(said|added|noted|wrote|explained|stressed)\b", lower)
                )
                if prev_ps and (cue_cond or has_speech):
                    speaker_candidates: List["spacy.tokens.Span"] = []
                    for p in prev_ps:
                        for tok in prev_sent:
                            if tok.lemma_ in SPEECH_VERBS and tok.pos_ in {"VERB", "AUX"}:
                                if any(
                                    ch.dep_ in {"nsubj", "nsubjpass"} and p.start <= ch.i < p.end
                                    for ch in tok.children
                                ):
                                    speaker_candidates.append(p)
                                    break
                    best_p = speaker_candidates[-1] if speaker_candidates else prev_ps[-1]
                    link_source = "prev_sent"

            if best_p is None and nc_for_role is not None and nc_for_role in nc_to_person:
                best_p = nc_to_person[nc_for_role]
                link_source = "intra_sent"

            if best_p is not None and nc_for_role is not None:
                nc_to_person[nc_for_role] = best_p

            org = organization_from_role_context(sent, role_span)
            if org is None:
                org = fallback_sentence_org(sent, role_span, best_p)
            org = normalize_ws(org) if org else "Unknown"
            org = title_case_org(org) if org != "Unknown" else "Unknown"

            org_type, is_media_org = classify_org_type(org)
            role_text = normalize_ws(role_span.text) or ""
            role_low = role_text.lower()

            person_txt = None
            if best_p is not None:
                person_txt = strip_leading_titles(best_p.text)
                person_txt = canonicalize_person_name(person_txt, surname_memory)

            person_value = person_txt if person_txt is not None else "Unknown"

            role_class = classify_role_class(role_text, org_type)
            is_journalist = int((role_class == "media_actor") or bool(is_media_org))
            is_government_official = int(role_class == "government_official")

            domain, domain_source, domain_certainty = infer_domain_from_role(role_text)

            low_sent = sent_text.lower()
            has_expert_keyword = int(any(k in low_sent for k in EXPERT_KEYWORDS))
            has_policy_keyword = int(any(k in low_sent for k in POLICY_KEYWORDS))

            cues = detect_country_cues(org if org != "Unknown" else "", sent, best_p)
            mention_country, mention_country_conf, mention_country_sources, mention_country_evidence, mention_country_ambiguous, mention_country_margin = decide_mention_country(cues)

            link_evidence, link_evidence_details = link_evidence_type(sent, role_span, best_p, link_source)

            expert_id = "Unknown"
            expert_profile_id = "Unknown"
            if person_value != "Unknown":
                person_key = normalize_for_id(person_value)
                expert_id = hash_id("exp", person_key, n=12)
                profile_key = "|".join([person_key, normalize_for_id(org or "")])
                expert_profile_id = hash_id("prof", profile_key, n=12)

            no_person = (person_value == "Unknown")
            no_keep_word = not any(k in role_low for k in ROLE_KEEP_KEYWORDS)
            if no_person and no_keep_word:
                continue

            row = {
                "extraction_version": EXTRACTION_VERSION,
                "row_id": row_id,
                "DateParsed": date,
                "expert_id": expert_id,
                "expert_profile_id": expert_profile_id,
                "sentence": sent_text,
                "role": role_text,
                "person": person_value,
                "organization": org,
                "org_type": org_type,
                "role_class": role_class,
                "is_media_org": int(is_media_org),
                "is_journalist": is_journalist,
                "is_government_official": is_government_official,
                "domain": domain,
                "domain_source": domain_source,
                "domain_certainty": domain_certainty,
                "link_source": link_source,
                "link_evidence": link_evidence,
                "link_evidence_details": link_evidence_details,
                "country_candidates_json": json.dumps(cues, ensure_ascii=False),
                "mention_country": mention_country,
                "mention_country_confidence": float(mention_country_conf),
                "mention_country_sources": mention_country_sources,
                "mention_country_evidence": mention_country_evidence,
                "mention_country_ambiguous": int(mention_country_ambiguous),
                "mention_country_margin": float(mention_country_margin),
                "experts_in_sentence_count": int(experts_in_sentence_count),
                "sentence_has_multiple_experts": int(sentence_has_multiple_experts),
                "has_expert_keyword": has_expert_keyword,
                "has_policy_keyword": has_policy_keyword,
            }
            rows.append(row)

    return rows


# =========================
# Expert-level canonical country
# =========================
def aggregate_expert_countries(
    mentions_df: pd.DataFrame,
    min_mentions: int = 1
) -> pd.DataFrame:
    df = mentions_df.copy()
    df = df[df["expert_id"].notna() & (df["expert_id"] != "Unknown")].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "expert_id", "canonical_country", "canonical_country_score",
            "canonical_country_ambiguous", "canonical_country_margin",
            "n_mentions", "n_distinct_orgs"
        ])

    def weight_row(r):
        w = float(r.get("mention_country_confidence", 0.0) or 0.0)
        src = str(r.get("mention_country_sources", ""))
        if "org_lookup" in src:
            w += 0.10
        if "syntax_of_from" in src:
            w += 0.05
        return min(1.2, w)

    df["__w"] = df.apply(weight_row, axis=1)

    agg = (
        df[df["mention_country"].notna() & (df["mention_country"] != "Unknown")]
        .groupby(["expert_id", "mention_country"], as_index=False)["__w"]
        .sum()
        .rename(columns={"__w": "country_score"})
    )

    experts = df.groupby("expert_id", as_index=False).agg(
        n_mentions=("expert_id", "size"),
        n_distinct_orgs=("organization", pd.Series.nunique)
    )

    agg_sorted = agg.sort_values(["expert_id", "country_score"], ascending=[True, False])
    top = agg_sorted.groupby("expert_id").head(2).copy()

    def compute_top(row_block: pd.DataFrame):
        row_block = row_block.sort_values("country_score", ascending=False)
        best = row_block.iloc[0]
        second_score = float(row_block.iloc[1]["country_score"]) if len(row_block) > 1 else 0.0
        margin = float(best["country_score"]) - second_score
        ambiguous = int((len(row_block) > 1 and margin <= 0.15) or float(best["country_score"]) < 0.8)
        return pd.Series({
            "canonical_country": best["mention_country"],
            "canonical_country_score": float(best["country_score"]),
            "canonical_country_margin": float(margin),
            "canonical_country_ambiguous": ambiguous,
        })

    canon = top.groupby("expert_id").apply(compute_top).reset_index()
    out = experts.merge(canon, on="expert_id", how="left")
    out["canonical_country"] = out["canonical_country"].fillna("Unknown")
    out["canonical_country_score"] = out["canonical_country_score"].fillna(0.0)
    out["canonical_country_margin"] = out["canonical_country_margin"].fillna(0.0)
    out["canonical_country_ambiguous"] = out["canonical_country_ambiguous"].fillna(1).astype(int)
    out.loc[out["n_mentions"] < min_mentions, ["canonical_country", "canonical_country_score"]] = ["Unknown", 0.0]
    out.loc[out["n_mentions"] < min_mentions, "canonical_country_ambiguous"] = 1
    return out


# =========================
# Role list helpers
# =========================
def expand_role_variants(roles: List[str]) -> List[str]:
    out = set()
    for r in roles:
        r = (r or "").strip()
        if not r:
            continue
        out.add(r)
        low = r.lower()
        if not low.endswith("s"):
            out.add(r + "s")
        if low.endswith("y") and len(low) > 2:
            out.add(r[:-1] + "ies")
        if low.endswith("ist"):
            out.add(r + "s")
    return sorted(out)


# =========================
# NLP pipeline setup
# =========================
def build_pipeline():
    nlp = spacy.load(NER_MODEL)
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    return nlp


# =========================
# Main
# =========================
def main():
    global ROLE_SURFACES_LOWER

    print(f"Reading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8", dtype="string")

    if TEXT_COL not in df.columns:
        raise KeyError(f"Column '{TEXT_COL}' not found. Available: {list(df.columns)}")

    # Normalize DateParsed — already named correctly, but format may vary
    if DATE_COL not in df.columns:
        print(f"Column '{DATE_COL}' not found. Using 'Unknown'.")
        df["DateParsed"] = "Unknown"
    else:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
        df["DateParsed"] = dt.dt.date.astype("string").fillna(df[DATE_COL])

    # Build role list from predefined lexicon (no key_role column in Izvestia dataset)
    role_list = expand_role_variants(PREDEFINED_ROLES)
    ROLE_SURFACES_LOWER = {r.lower() for r in role_list}

    subset = (
        df.loc[df[TEXT_COL].notna(), [TEXT_COL, "DateParsed"]]
          .reset_index(drop=False)
          .rename(columns={"index": "row_id"})
    )

    total_docs = len(subset)
    print(f"Processing {total_docs} articles.")
    print(f"Role lexicon size (incl. variants): {len(role_list)}")
    print(f"Extraction version: {EXTRACTION_VERSION}")

    nlp = build_pipeline()
    matcher = build_phrase_matcher(nlp, role_list)

    BATCH_SIZE = 8
    N_PROCESS = 1
    CHUNK_SIZE = 500  # smaller than Lenta since dataset is ~1350 articles

    print("Starting expert extraction...")
    all_rows: List[Dict[str, Any]] = []
    out_path_temp = OUT_PATH.with_suffix(".partial.csv")
    experts_out_temp = OUT_EXPERTS_PATH.with_suffix(".partial.csv")

    for start in range(0, total_docs, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_docs)
        chunk = subset.iloc[start:end]
        print(f"\nChunk {start}-{end - 1} (size {len(chunk)})")

        rows_chunk: List[Dict[str, Any]] = []
        docs = nlp.pipe(chunk[TEXT_COL].tolist(), batch_size=BATCH_SIZE, n_process=N_PROCESS)

        for rec, doc in tqdm(
            zip(chunk.to_dict("records"), docs),
            total=len(chunk),
            desc="Extracting experts",
            leave=False,
        ):
            rows_chunk.extend(
                extract_expert_rows_from_doc(
                    doc=doc,
                    row_id=int(rec["row_id"]),
                    date=str(rec["DateParsed"]),
                    matcher=matcher,
                )
            )

        all_rows.extend(rows_chunk)

        tmp_df = pd.DataFrame(all_rows)
        if not tmp_df.empty:
            tmp_df.to_csv(out_path_temp, index=False, encoding="utf-8")
            print(f"Saved partial mentions: {len(tmp_df)} rows")
            experts_df = aggregate_expert_countries(tmp_df)
            experts_df.to_csv(experts_out_temp, index=False, encoding="utf-8")

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved final mentions to {OUT_PATH}")
    print(f"Total expert mentions: {len(out)}")

    experts_df = aggregate_expert_countries(out)
    experts_df.to_csv(OUT_EXPERTS_PATH, index=False, encoding="utf-8")
    print(f"Saved expert-level table to {OUT_EXPERTS_PATH}")
    print(f"Total experts: {len(experts_df)}")

    with pd.option_context("display.max_columns", None, "display.max_colwidth", 140):
        print("\nMentions preview:")
        print(out.head(10))
        print("\nExperts preview:")
        print(experts_df.head(10))


if __name__ == "__main__":
    main()
