from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set

import pandas as pd

from src.features.keyword_expansion import tokenize_sentence
from src.features.rule_based_topic_classifier import TOPIC_RULES


STRONG_CONTEXT_TOKENS = {
    "oil",
    "crude",
    "petroleum",
    "pipeline",
    "pipelines",
    "refinery",
    "refineries",
    "saudi",
    "aramco",
    "opec",
    "shale",
    "wti",
    "brent",
    "inventory",
    "inventories",
    "stockpile",
    "stockpiles",
    "barrels",
    "gasoline",
    "diesel",
    "jet",
    "fuel",
    "spr",
    "tanker",
    "shipping",
}

SECONDARY_CONTEXT_TOKENS = {
    "supply",
    "demand",
    "production",
    "output",
    "export",
    "exports",
    "drilling",
    "sanctions",
    "iran",
    "russia",
    "war",
    "conflict",
    "fed",
    "dollar",
    "index",
    "inflation",
    "recession",
    "rates",
    "rate",
    "hike",
    "hikes",
    "cut",
    "cuts",
    "macro",
}

CONTEXT_TOKENS = STRONG_CONTEXT_TOKENS | SECONDARY_CONTEXT_TOKENS

DIRECTIONAL_SIGNAL_TOKENS = {
    "draw",
    "draws",
    "build",
    "builds",
    "cut",
    "cuts",
    "cutback",
    "curb",
    "curbs",
    "boost",
    "boosts",
    "increase",
    "increases",
    "raise",
    "raises",
    "ramp",
    "ramps",
    "disruption",
    "disruptions",
    "outage",
    "outages",
    "shutdown",
    "shutdowns",
    "restart",
    "restarts",
    "resume",
    "resumes",
    "recovery",
    "recoveries",
    "surge",
    "surges",
    "growth",
    "shortage",
    "shortages",
    "sanction",
    "sanctions",
    "conflict",
    "war",
    "attack",
    "attacks",
    "hike",
    "hikes",
    "cuts",
    "inflation",
    "recession",
    "slowdown",
    "weakness",
    "decline",
    "declines",
    "drop",
    "drops",
    "fall",
    "falls",
    "rise",
    "rises",
}

NOISE_TOKENS = {
    "article",
    "generated",
    "benzinga",
    "automated",
    "engine",
    "reviewed",
    "editor",
    "editors",
    "images",
    "image",
    "story",
    "png",
    "alt",
    "thead",
    "tbody",
    "conference",
    "calls",
    "financial",
    "statements",
    "target",
    "targets",
    "average",
    "strong",
    "bullish",
    "bearish",

    # url / html / link residue
    "href",
    "http",
    "https",
    "www",
    "com",
    "html",
    "src",
    "link",
    "img",
    "nbsp",

    # market widget / exchange residue
    "stock",
    "stocks",
    "nyse",
    "nasdaq",
    "ticker",
    "shares",
    "quote",
    "quotes",
    "analyst",
    "analysts",
    "rating",
    "ratings",
    "sector",
    "etf",

    # page-chrome / site furniture
    "newsletter",
    "signup",
    "login",
    "copyright",
    "related",
    "watch",
    "video",
    "news",
    "widget",
    "table",
}

ALLOWED_SHORT_TOKENS = {
    "oil",
    "wti",
    "spr",
    "fed",
    "api",
    "eia",
    "lng",
    "opec",
}

PHRASE_REJECT_PATTERNS = [
    re.compile(r"\b(?:href|http|https|www|html|src|img|png|nbsp)\b"),
    re.compile(r"\b(?:nyse|nasdaq|ticker|shares?|quote|quotes?)\b"),
    re.compile(r"\b(?:analyst|analysts|rating|ratings|sector|etf|widget|newsletter|signup|login)\b"),
    re.compile(r"\b(?:stock|stocks)\s+[a-z]{2,5}\b"),
    re.compile(r"\b[a-z]{2,5}\s+(?:nyse|nasdaq)\b"),
    re.compile(r"\b(?:com|www)\s+[a-z]{2,5}\b"),
]


def existing_seed_phrases() -> Set[str]:
    seeds: Set[str] = set()
    for cfg in TOPIC_RULES.values():
        for field in ("core_terms", "direction_terms", "fixed_phrases"):
            for phrase in cfg.get(field, []):
                normalized = " ".join(tokenize_sentence(phrase))
                if normalized:
                    seeds.add(normalized)
    return seeds


def has_strong_domain_anchor(window: List[str]) -> bool:
    return any(token in STRONG_CONTEXT_TOKENS for token in window)


def has_sufficient_domain_context(window: List[str]) -> bool:
    strong_hits = sum(token in STRONG_CONTEXT_TOKENS for token in window)
    secondary_hits = sum(token in SECONDARY_CONTEXT_TOKENS for token in window)
    return strong_hits >= 1 or secondary_hits >= 2


def has_directional_signal(window: List[str]) -> bool:
    return any(token in DIRECTIONAL_SIGNAL_TOKENS for token in window)


def looks_like_junk_fragment(window: List[str]) -> bool:
    short_suspicious = [
        token
        for token in window
        if len(token) <= 4 and token not in ALLOWED_SHORT_TOKENS
    ]
    return len(short_suspicious) >= max(2, len(window) - 1)


def reject_candidate_window(window: List[str]) -> bool:
    if any(token in NOISE_TOKENS for token in window):
        return True

    phrase = " ".join(window)
    if any(pattern.search(phrase) for pattern in PHRASE_REJECT_PATTERNS):
        return True

    if looks_like_junk_fragment(window):
        return True

    if not has_sufficient_domain_context(window):
        return True

    if not has_directional_signal(window):
        return True

    return False


def generate_candidate_ngrams(text: str, max_n: int = 3) -> Set[str]:
    tokens = tokenize_sentence(text)
    phrases: Set[str] = set()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            if reject_candidate_window(window):
                continue
            phrase = " ".join(window)
            if len(phrase.replace(" ", "")) < 6:
                continue
            phrases.add(phrase)
    return phrases


def mine_directional_phrases(
    articles: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "direction_label",
    topics_col: str = "rule_based_topics",
    min_doc_freq: int = 6,
    max_n: int = 3,
    top_n_per_label: int = 250,
) -> pd.DataFrame:
    """
    Mine candidate directional phrases from bullish vs bearish classified articles.

    Phrases are ranked by smoothed log-odds between bullish and bearish document
    frequencies, with existing rule seeds excluded.
    """
    if articles.empty:
        return pd.DataFrame(
            columns=[
                "phrase",
                "label",
                "log_odds",
                "bullish_count",
                "bearish_count",
                "total_count",
                "bullish_share",
                "bearish_share",
                "dominant_topics",
            ]
        )

    df = articles.copy()
    df = df[df[label_col].isin(["bullish", "bearish"])].copy()
    if df.empty:
        return pd.DataFrame()

    bullish_docs = int((df[label_col] == "bullish").sum())
    bearish_docs = int((df[label_col] == "bearish").sum())
    if bullish_docs == 0 or bearish_docs == 0:
        return pd.DataFrame()

    seed_phrases = existing_seed_phrases()
    bullish_counts: Counter[str] = Counter()
    bearish_counts: Counter[str] = Counter()
    phrase_topic_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for _, row in df.iterrows():
        text = str(row.get(text_col, "") or "")
        if not text.strip():
            continue

        phrases = generate_candidate_ngrams(text, max_n=max_n)
        phrases = {phrase for phrase in phrases if phrase not in seed_phrases}
        if not phrases:
            continue

        topics = [
            topic.strip()
            for topic in str(row.get(topics_col, "") or "").split(",")
            if topic.strip()
        ]

        target = bullish_counts if row[label_col] == "bullish" else bearish_counts
        for phrase in phrases:
            target[phrase] += 1
            for topic in topics:
                phrase_topic_counts[phrase][topic] += 1

    records: List[Dict[str, object]] = []
    vocab = set(bullish_counts) | set(bearish_counts)
    for phrase in vocab:
        bull = bullish_counts.get(phrase, 0)
        bear = bearish_counts.get(phrase, 0)
        total = bull + bear
        if total < min_doc_freq:
            continue

        bull_share = (bull + 1.0) / (bullish_docs + 2.0)
        bear_share = (bear + 1.0) / (bearish_docs + 2.0)
        log_odds = math.log(bull_share / bear_share)
        if math.isclose(log_odds, 0.0, abs_tol=1e-9):
            continue

        label = "bullish" if log_odds > 0 else "bearish"
        dominant_topics = ",".join(
            topic
            for topic, _ in phrase_topic_counts[phrase].most_common(3)
        )
        records.append(
            {
                "phrase": phrase,
                "label": label,
                "log_odds": round(log_odds, 6),
                "bullish_count": bull,
                "bearish_count": bear,
                "total_count": total,
                "bullish_share": round(bull_share, 6),
                "bearish_share": round(bear_share, 6),
                "dominant_topics": dominant_topics,
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    bullish = (
        out[out["label"] == "bullish"]
        .sort_values(["log_odds", "total_count"], ascending=[False, False])
        .head(top_n_per_label)
    )
    bearish = (
        out[out["label"] == "bearish"]
        .assign(abs_log_odds=lambda x: x["log_odds"].abs())
        .sort_values(["abs_log_odds", "total_count"], ascending=[False, False])
        .drop(columns=["abs_log_odds"])
        .head(top_n_per_label)
    )
    return (
        pd.concat([bullish, bearish], ignore_index=True)
        .sort_values(["label", "total_count", "phrase"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
