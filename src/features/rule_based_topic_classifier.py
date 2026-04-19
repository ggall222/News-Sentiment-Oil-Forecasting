from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.sentiment_indicators import compute_weekly_indicators_long

# =========================
# Configuration
# =========================

TOPIC_RULES: Dict[str, Dict[str, Any]] = {
    "inventory_draw": {
        "core_terms": [
            "inventory", "inventories", "stock", "stocks", "stockpile", "stockpiles",
            "crude stocks", "crude inventories", "gasoline stocks", "distillate stocks",
            "fuel inventories", "crude stockpiles", "stores", "reserves",
        ],
        "direction_terms": [
            "draw", "drawdown", "fall", "fell", "drop", "dropped", "decline",
            "declined", "tightening", "depletion", "drain", "reduction",
        ],
        "fixed_phrases": [
            "inventory draw", "crude draw", "stock draw", "stockpile draw",
            "unexpected draw", "surprise draw", "inventory depletion",
            "tight inventories", "record draw",
        ],
        "base_weight": 1.00,
        "headline_bonus": 0.15,
        "surprise_bonus": 0.30,
    },
    "inventory_build": {
        "core_terms": [
            "inventory", "inventories", "stock", "stocks", "stockpile", "stockpiles",
            "crude stocks", "crude inventories", "gasoline stocks", "distillate stocks",
            "fuel inventories", "crude stockpiles", "stores", "reserves",
        ],
        "direction_terms": [
            "build", "rise", "rose", "increase", "increased", "growth", "accumulation",
            "surge", "surged", "climb", "climbed", "expansion",
        ],
        "fixed_phrases": [
            "inventory build", "crude build", "stock build", "stockpile build",
            "unexpected build", "surprise build", "record build",
            "rising inventories", "surging inventories",
        ],
        "base_weight": 1.00,
        "headline_bonus": 0.15,
        "surprise_bonus": 0.30,
    },
    "opec_cut": {
        "core_terms": ["opec", "opec+", "saudi", "aramco", "russia", "oil producers", "producer alliance"],
        "direction_terms": [
            "cut", "cuts", "curb", "curbs", "reduce", "reduces", "reduction",
            "tighten", "tightens", "restrain", "restrains", "withhold", "withholds",
            "extends cuts", "deepens cuts", "voluntary cuts",
        ],
        "fixed_phrases": [
            "opec cuts", "opec production cut", "opec output cut", "opec+ cuts",
            "saudi voluntary cut", "opec extends cuts", "opec deeper cuts",
        ],
        "base_weight": 1.10,
        "headline_bonus": 0.15,
        "surprise_bonus": 0.20,
    },
    "opec_output_increase": {
        "core_terms": ["opec", "opec+", "saudi", "russia", "producer alliance"],
        "direction_terms": [
            "increase", "increases", "boost", "boosts", "raise", "raises",
            "ramp", "ramps", "higher output", "more output", "more barrels",
            "return barrels", "restore supply",
        ],
        "fixed_phrases": [
            "opec boosts output", "opec increases production", "opec raises quotas",
            "opec raises output", "saudi production increase",
            "returns barrels to market",
        ],
        "base_weight": 1.10,
        "headline_bonus": 0.15,
        "surprise_bonus": 0.20,
    },
    "supply_disruption": {
        "core_terms": [
            "supply", "production", "output", "exports", "export", "pipeline",
            "terminal", "field", "platform", "facility", "facilities",
        ],
        "direction_terms": [
            "disruption", "outage", "halt", "shutdown", "offline", "curtailment",
            "curbed", "loss", "losses", "shortage", "interruption", "sabotage",
            "attack", "damaged", "force majeure", "shut in",
        ],
        "fixed_phrases": [
            "supply disruption", "production outage", "pipeline shutdown",
            "export disruption", "force majeure", "barrels offline",
            "production halt", "output halt",
        ],
        "base_weight": 1.15,
        "headline_bonus": 0.20,
        "surprise_bonus": 0.20,
    },
    "supply_recovery": {
        "core_terms": [
            "supply", "production", "output", "exports", "pipeline",
            "terminal", "field", "platform", "facility", "stores", "reserves"
        ],
        "direction_terms": [
            "restart", "restarts", "resume", "resumes", "restore", "restores",
            "reopen", "reopens", "release","recover", "recovers", "returns", "flows resume",
        ],
        "fixed_phrases": [
            "pipeline restart", "production resumes", "supply returns",
            "exports resume", "output restored",
        ],
        "base_weight": 1.00,
        "headline_bonus": 0.10,
        "surprise_bonus": 0.10,
    },
    "refinery_outage": {
        "core_terms": [
            "refinery", "refining","cracker", "processing unit", "plant", "throughput", "runs",
        ],
        "direction_terms": [
            "outage", "shutdown", "fire", "explosion", "offline", "maintenance",
            "unplanned maintenance", "damage", "downtime", "halted", "cut", "cuts",
        ],
        "fixed_phrases": [
            "refinery outage", "refinery shutdown", "refinery fire", "refinery explosion",
            "cracker outage", "unplanned maintenance", "run cuts",
        ],
        "base_weight": 1.05,
        "headline_bonus": 0.10,
        "surprise_bonus": 0.10,
    },
    "refinery_restart": {
        "core_terms": ["refinery", "cracker", "processing unit", "plant", "throughput", "runs"],
        "direction_terms": [
            "restart", "restarts", "resume", "resumes", "reopen", "reopens",
            "recovery", "restored", "back online",
        ],
        "fixed_phrases": [
            "refinery restart", "refinery resumes", "back online", "operations resume",
        ],
        "base_weight": 0.95,
        "headline_bonus": 0.08,
        "surprise_bonus": 0.08,
    },
    "geopolitical_risk": {
        "core_terms": [
            "sanctions", "embargo", "war", "conflict", "deploys", "invasion", "troops", "tensions", "attack",
            "drone", "military", "retaliation", "hostilities", "hormuz",
            "red sea", "gulf", "shipping", "tanker",
        ],
        "direction_terms": [
            "escalate", "escalates", "rise", "rises", "threat", "threatens",
            "disrupt", "disrupts", "attack", "attacks", "military action", "strike", "strikes",
            "risk", "risks", "sanctions", "embargo", 
        ],
        "fixed_phrases": [
            "middle east tensions", "strait of hormuz tensions", "pipeline attack",
            "drone strike oil facility", "shipping attacks", "tanker attacks",
            "oil embargo", "new sanctions",
        ],
        "base_weight": 1.40,
        "headline_bonus": 0.20,
        "surprise_bonus": 0.10,
    },
    "geopolitical_deescalation": {
        "core_terms": [
            "ceasefire", "negotiation", "negotiations", "truce", "diplomacy", "talks", "agreement", "deal",
            "sanctions relief", "waiver", "de-escalation",
        ],
        "direction_terms": [
            "ease", "eases", "easing", "resolve", "resolved", "stabilize",
            "stabilized", "calm", "calms",
        ],
        "fixed_phrases": [
            "sanctions relief", "ceasefire talks", "tensions ease", "conflict de-escalation",
        ],
        "base_weight": 1.10,
        "headline_bonus": 0.10,
        "surprise_bonus": 0.08,
    },
    "shipping_disruption": {
        "core_terms": [
            "shipping", "tanker", "tankers", "port", "transit", "marine", "maritime",
            "route", "routes", "red sea", "hormuz",
        ],
        "direction_terms": [
            "disruption", "delay", "delays", "rerouting", "closure", "closed",
            "attack", "attacks", "bottleneck", "congestion", "halt", "risk",
        ],
        "fixed_phrases": [
            "shipping disruption", "tanker rerouting", "port closure",
            "shipping bottlenecks", "maritime attacks", "transit bottlenecks",
        ],
        "base_weight": 1.15,
        "headline_bonus": 0.15,
        "surprise_bonus": 0.10,
    },
    "demand_strength": {
        "core_terms": [
            "demand", "consumption", "travel", "driving", "jet fuel",
            "aviation", "industrial activity", "manufacturing",
        ],
        "direction_terms": [
            "surge", "rebound", "recovery", "rise", "rises", "increase",
            "increases", "growth", "strong", "improves", "boom",
        ],
        "fixed_phrases": [
            "strong oil demand", "demand surge", "demand rebound", "travel demand surge",
            "summer driving demand", "aviation demand rebound", "consumption recovery",
        ],
        "base_weight": 0.95,
        "headline_bonus": 0.10,
        "surprise_bonus": 0.05,
    },
    "demand_weakness": {
        "core_terms": [
            "demand", "consumption", "travel", "driving", "jet fuel",
            "aviation", "industrial activity", "manufacturing", "economy",
        ],
        "direction_terms": [
            "weakens", "weak", "slowdown", "drops", "drop", "falls", "fall",
            "declines", "decline", "destruction", "softens", "contraction",
            "recession", "slump",
        ],
        "fixed_phrases": [
            "demand destruction", "weak fuel demand", "demand slowdown",
            "travel demand weakens", "jet fuel demand drops", "recession fears",
        ],
        "base_weight": 1.00,
        "headline_bonus": 0.10,
        "surprise_bonus": 0.10,
    },
    "macro_financial": {
        "core_terms": [
            "dollar", "fed", "interest rates", "rate cuts", "rate hike", "inflation",
            "recession", "macro", "financial conditions", "risk assets",
        ],
        "direction_terms": [
            "tightens", "eases", "strengthens", "weakens", "cuts", "hikes",
            "pressures", "supports",
        ],
        "fixed_phrases": [
            "stronger dollar", "fed tightening", "rate cuts", "recession fears",
        ],
        "base_weight": 0.90,
        "headline_bonus": 0.08,
        "surprise_bonus": 0.05,
    },
}

SURPRISE_MODIFIERS = {
    "unexpected": 0.25,
    "surprise": 0.25,
    "alarming": 0.30,
    "larger-than-expected": 0.30,
    "smaller-than-expected": 0.30,
    "unexpectedly": 0.25,
    "more-than-expected": 0.20,
    "less-than-expected": 0.20,
}


TOPIC_DIRECTIONS: Dict[str, int] = {
    "inventory_draw": 1,
    "inventory_build": -1,
    "opec_cut": 1,
    "opec_output_increase": -1,
    "supply_disruption": 1,
    "supply_recovery": -1,
    "refinery_outage": -1,
    "refinery_restart": 1,
    "geopolitical_risk": 1,
    "geopolitical_deescalation": -1,
    "shipping_disruption": 1,
    "demand_strength": 1,
    "demand_weakness": -1,
    "macro_financial": -1,
}

POSITIVE_MACRO_TERMS = {
    "rate cuts",
    "cuts",
    "eases",
    "easing",
    "weakens",
    "weakening",
    "supports",
    "support",
}

NEGATIVE_MACRO_TERMS = {
    "tightens",
    "tightening",
    "strengthens",
    "stronger dollar",
    "hikes",
    "hike",
    "pressures",
    "pressure",
    "recession",
    "inflation",
}

PROMOTABLE_RULE_FIELDS = {"fixed_phrases", "direction_terms", "core_terms"}
DEFAULT_RULE_OVERRIDE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "lexicons"
    / "rule_based_phrase_overrides.json"
)

MAGNITUDE_MODIFIERS = {
    "record": 0.25,
    "massive": 0.20,
    "major": 0.18,
    "considerable": 0.15,
    "considerably": 0.15,
    "consequential": 0.18,
    "sharp": 0.12,
    "steep": 0.12,
    "severe": 0.18,
    "sudden": 0.12,
    "big": 0.08,
    "sizable": 0.10,
    "largest": 0.20,
    "deep": 0.12,
}

PERSISTENCE_MODIFIERS = {
    "ongoing": 0.08,
    "escalating": 0.15,
    "worsening": 0.15,
    "prolonged": 0.12,
    "extended": 0.10,
    "renewed": 0.10,
}

NEGATIONS = {
    "no", "not", "denies", "deny", "without", "unlikely", "avoids", "avoid",
    "fails to", "failed to",
}

SOFTENERS = {
    "temporary", "short-lived", "limited impact", "contained disruption",
    "modest", "partially offset", "largely offset", "exports remain unaffected",
    "supply unaffected",
}


# =========================
# Helpers
# =========================

def normalize_text(text: Optional[str]) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9+\-/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compile_phrase_patterns(phrases: Sequence[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    for phrase in phrases:
        escaped = re.escape(phrase).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
        patterns.append((phrase, pattern))
    return patterns


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text) for p in phrases)


def find_term_positions(text: str, terms: Sequence[str]) -> List[Tuple[str, int, int]]:
    matches: List[Tuple[str, int, int]] = []
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        for m in pattern.finditer(text):
            matches.append((term, m.start(), m.end()))
    return matches


def near(a_start: int, b_start: int, max_distance: int = 60) -> bool:
    return abs(a_start - b_start) <= max_distance


def local_modifier_bonus(text: str, start: int, end: int, window: int = 45) -> Tuple[float, bool]:
    segment = text[max(0, start - window): min(len(text), end + window)]
    bonus = 0.0
    surprise_found = False

    for word, val in SURPRISE_MODIFIERS.items():
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", segment):
            bonus += val
            surprise_found = True

    for word, val in MAGNITUDE_MODIFIERS.items():
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", segment):
            bonus += val

    for word, val in PERSISTENCE_MODIFIERS.items():
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", segment):
            bonus += val

    for word in SOFTENERS:
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", segment):
            bonus -= 0.15

    return max(-0.25, min(bonus, 0.60)), surprise_found


def is_negated(text: str, start: int, window: int = 25) -> bool:
    segment = text[max(0, start - window): start]
    return any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", segment) for word in NEGATIONS)


def softmax_dict(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=float)
    if np.allclose(vals.sum(), 0):
        return {k: 0.0 for k in keys}
    exp_vals = np.exp(vals - vals.max())
    probs = exp_vals / exp_vals.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


@dataclass
class MatchRecord:
    topic: str
    source: str
    matched_text: str
    start: int
    end: int
    weight: float
    surprise_flag: int


# =========================
# Classifier
# =========================


def _copy_topic_rules(topic_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    copied: Dict[str, Dict[str, Any]] = {}
    for topic, cfg in topic_rules.items():
        copied[topic] = {}
        for key, value in cfg.items():
            copied[topic][key] = list(value) if isinstance(value, list) else value
    return copied


def _load_topic_rule_overrides(
    topic_rules: Dict[str, Dict[str, Any]],
    overrides_path: Optional[Path | str] = None,
) -> Dict[str, Dict[str, Any]]:
    merged = _copy_topic_rules(topic_rules)
    path = Path(overrides_path) if overrides_path is not None else DEFAULT_RULE_OVERRIDE_PATH
    if not path.exists():
        return merged

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return merged

    topics_payload = payload.get("topics", payload)
    if not isinstance(topics_payload, dict):
        return merged

    for topic, field_map in topics_payload.items():
        if topic not in merged or not isinstance(field_map, dict):
            continue
        for field, phrases in field_map.items():
            if field not in PROMOTABLE_RULE_FIELDS or not isinstance(phrases, list):
                continue
            existing = {
                str(item).strip().lower()
                for item in merged[topic].get(field, [])
            }
            merged[topic].setdefault(field, [])
            for phrase in phrases:
                normalized = str(phrase).strip().lower()
                if normalized and normalized not in existing:
                    merged[topic][field].append(normalized)
                    existing.add(normalized)

    return merged

class RuleBasedOilTopicClassifier:
    def __init__(
        self,
        topic_rules: Optional[Dict[str, Dict[str, Any]]] = None,
        approved_phrases_path: Optional[Path | str] = None,
        headline_weight: float = 1.25,
        body_weight: float = 1.00,
        match_window: int = 60,
        min_topic_score: float = 0.75,
    ) -> None:
        base_rules = topic_rules or TOPIC_RULES
        self.topic_rules = _load_topic_rule_overrides(base_rules, approved_phrases_path)
        self.headline_weight = headline_weight
        self.body_weight = body_weight
        self.match_window = match_window
        self.min_topic_score = min_topic_score

        self.fixed_phrase_patterns = {
            topic: compile_phrase_patterns(cfg.get("fixed_phrases", []))
            for topic, cfg in self.topic_rules.items()
        }

    def _score_source(self, text: str, source_name: str, source_weight: float) -> Tuple[Dict[str, float], List[MatchRecord]]:
        topic_scores: Dict[str, float] = {topic: 0.0 for topic in self.topic_rules}
        match_records: List[MatchRecord] = []

        for topic, cfg in self.topic_rules.items():
            base_weight = float(cfg.get("base_weight", 1.0))
            headline_bonus = float(cfg.get("headline_bonus", 0.0)) if source_name == "headline" else 0.0
            surprise_bonus = float(cfg.get("surprise_bonus", 0.0))

            # Fixed phrase hits
            for phrase, pattern in self.fixed_phrase_patterns[topic]:
                for m in pattern.finditer(text):
                    if is_negated(text, m.start()):
                        continue
                    bonus, surprise_found = local_modifier_bonus(text, m.start(), m.end())
                    weight = (base_weight + headline_bonus + bonus + (surprise_bonus if surprise_found else 0.0)) * source_weight
                    weight = max(0.0, weight)
                    topic_scores[topic] += weight
                    match_records.append(
                        MatchRecord(topic, source_name, phrase, m.start(), m.end(), round(weight, 4), int(surprise_found))
                    )

            # Compositional rule hits: core_terms near direction_terms
            core_positions = find_term_positions(text, cfg.get("core_terms", []))
            dir_positions = find_term_positions(text, cfg.get("direction_terms", []))
            used_pairs = set()

            for core_term, c_start, c_end in core_positions:
                if is_negated(text, c_start):
                    continue
                for dir_term, d_start, d_end in dir_positions:
                    if is_negated(text, d_start):
                        continue
                    if not near(c_start, d_start, self.match_window):
                        continue

                    pair_key = (topic, min(c_start, d_start), max(c_end, d_end))
                    if pair_key in used_pairs:
                        continue
                    used_pairs.add(pair_key)

                    span_start = min(c_start, d_start)
                    span_end = max(c_end, d_end)
                    bonus, surprise_found = local_modifier_bonus(text, span_start, span_end)
                    weight = (base_weight + headline_bonus + bonus + (surprise_bonus if surprise_found else 0.0)) * source_weight
                    weight = max(0.0, weight)

                    matched_text = f"{core_term} + {dir_term}"
                    topic_scores[topic] += weight
                    match_records.append(
                        MatchRecord(topic, source_name, matched_text, span_start, span_end, round(weight, 4), int(surprise_found))
                    )

        # remove zero scores
        topic_scores = {k: round(v, 6) for k, v in topic_scores.items() if v > 0}
        return topic_scores, match_records

    def classify_article(
        self,
        headline: Optional[str],
        body_text: Optional[str] = None,
        published_at: Optional[str] = None,
        article_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        headline_norm = normalize_text(headline)
        body_norm = normalize_text(body_text)

        headline_scores, headline_matches = self._score_source(
            headline_norm, source_name="headline", source_weight=self.headline_weight
        )
        body_scores, body_matches = self._score_source(
            body_norm, source_name="body", source_weight=self.body_weight
        )

        all_topics = sorted(set(headline_scores) | set(body_scores))
        raw_scores: Dict[str, float] = {}
        for topic in all_topics:
            raw_scores[topic] = round(headline_scores.get(topic, 0.0) + body_scores.get(topic, 0.0), 6)

        confidences = softmax_dict(raw_scores)
        labels = [topic for topic, score in raw_scores.items() if score >= self.min_topic_score]
        top_label = max(raw_scores, key=raw_scores.get) if raw_scores else None

        match_df = pd.DataFrame([m.__dict__ for m in (headline_matches + body_matches)])
        if match_df.empty:
            topic_match_counts: Dict[str, int] = {}
            topic_surprise_counts: Dict[str, int] = {}
            strongest_phrases: Dict[str, Optional[str]] = {}
        else:
            topic_match_counts = match_df.groupby("topic").size().astype(int).to_dict()
            topic_surprise_counts = match_df.groupby("topic")["surprise_flag"].sum().astype(int).to_dict()
            strongest_idx = match_df.groupby("topic")["weight"].idxmax()
            strongest_phrases = {
                row["topic"]: row["matched_text"]
                for _, row in match_df.loc[strongest_idx].iterrows()
            }

        result: Dict[str, Any] = {
            "article_id": article_id,
            "published_at": published_at,
            "headline": headline,
            "body_text": body_text,
            "headline_norm": headline_norm,
            "body_norm": body_norm,
            "top_label": top_label,
            "labels": labels,
            "raw_scores": raw_scores,
            "confidences": confidences,
            "match_counts": topic_match_counts,
            "surprise_counts": topic_surprise_counts,
            "strongest_phrases": strongest_phrases,
            "matches_detail": match_df.to_dict(orient="records") if not match_df.empty else [],
        }

        # Flatten article-level features
        for topic in self.topic_rules.keys():
            raw = raw_scores.get(topic, 0.0)
            conf = confidences.get(topic, 0.0)
            count = topic_match_counts.get(topic, 0)
            surprise_count = topic_surprise_counts.get(topic, 0)
            strongest = strongest_phrases.get(topic)

            result[f"topic_{topic}"] = int(raw >= self.min_topic_score)
            result[f"{topic}_score"] = raw
            result[f"{topic}_confidence"] = conf
            result[f"{topic}_match_count"] = count
            result[f"{topic}_surprise_count"] = surprise_count
            result[f"{topic}_high_signal"] = int((count >= 2 and raw >= 1.5) or raw >= 2.0)
            result[f"{topic}_strongest_phrase"] = strongest

        return result

    def classify_articles(
        self,
        df: pd.DataFrame,
        headline_col: str = "headline",
        body_col: str = "body_text",
        date_col: str = "published_at",
        article_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            article_id = row[article_id_col] if article_id_col and article_id_col in df.columns else idx
            result = self.classify_article(
                headline=row.get(headline_col),
                body_text=row.get(body_col),
                published_at=row.get(date_col),
                article_id=article_id,
            )
            rows.append(result)

        return pd.DataFrame(rows)


def build_weekly_topic_features(
    article_features: pd.DataFrame,
    date_col: str = "published_at",
    freq: str = "W-FRI",
    decay_alpha: float = 0.6,
) -> pd.DataFrame:
    if article_features.empty:
        return pd.DataFrame()

    df = article_features.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    topic_names = [
        c[len("topic_"):] for c in df.columns
        if c.startswith("topic_")
    ]

    numeric_cols: List[str] = []
    agg_map: Dict[str, List[str]] = {}

    for topic in topic_names:
        for suffix in ["score", "confidence", "match_count", "surprise_count", "high_signal"]:
            col = f"{topic}_{suffix}"
            if col in df.columns:
                numeric_cols.append(col)
                if suffix in {"score", "confidence"}:
                    agg_map[col] = ["sum", "mean", "max", "var"]
                else:
                    agg_map[col] = ["sum", "mean", "max"]

        topic_flag = f"topic_{topic}"
        if topic_flag in df.columns:
            numeric_cols.append(topic_flag)
            agg_map[topic_flag] = ["sum", "mean"]

    grouped = df.set_index(date_col).groupby(pd.Grouper(freq=freq)).agg(agg_map)
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index().rename(columns={date_col: "date"})

    # article count
    article_count = (
        df.set_index(date_col)
        .groupby(pd.Grouper(freq=freq))
        .size()
        .rename("article_count")
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    grouped = grouped.merge(article_count, on="date", how="left")

    # share features
    for topic in topic_names:
        sum_col = f"topic_{topic}_sum"
        if sum_col in grouped.columns:
            grouped[f"{topic}_share"] = np.where(
                grouped["article_count"] > 0,
                grouped[sum_col] / grouped["article_count"],
                0.0,
            )

    # simple decays based on score_sum
    grouped = grouped.sort_values("date").reset_index(drop=True)
    for topic in topic_names:
        score_sum_col = f"{topic}_score_sum"
        if score_sum_col in grouped.columns:
            decay_vals = []
            prev = 0.0
            for x in grouped[score_sum_col].fillna(0.0).tolist():
                prev = decay_alpha * prev + x
                decay_vals.append(prev)
            grouped[f"{topic}_score_decay"] = decay_vals

    return grouped

def build_parquet_safe_article_features(
    article_features: pd.DataFrame,
    date_col: str = "published_at",
) -> pd.DataFrame:
    """
    Convert classifier output into a parquet-friendly article feature frame.
    """
    if article_features.empty:
        return article_features.copy()

    df = article_features.copy()
    if date_col in df.columns:
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "labels" in df.columns:
        df["labels_csv"] = df["labels"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else ""
        )

    json_cols = [
        "raw_scores",
        "confidences",
        "match_counts",
        "surprise_counts",
        "strongest_phrases",
        "matches_detail",
    ]
    for col in json_cols:
        if col in df.columns:
            df[f"{col}_json"] = df[col].apply(lambda x: json.dumps(x, sort_keys=True, default=str))

    drop_cols = [
        "headline_norm",
        "body_norm",
        "labels",
        *json_cols,
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def topic_direction_sign(topic: str, strongest_phrase: Optional[str] = None) -> int:
    if topic != "macro_financial":
        return int(TOPIC_DIRECTIONS.get(topic, 0))

    phrase = normalize_text(strongest_phrase or "")
    if any(term in phrase for term in POSITIVE_MACRO_TERMS):
        return 1
    if any(term in phrase for term in NEGATIVE_MACRO_TERMS):
        return -1
    return int(TOPIC_DIRECTIONS["macro_financial"])


def build_directional_article_features(article_features: pd.DataFrame) -> pd.DataFrame:
    """
    Add article-level directional sentiment fields derived from rule-based topics.

    `rule_based_sentiment` is the signed confidence mass across matched topics,
    producing a bounded value in [-1, 1].
    """
    if article_features.empty:
        return article_features.copy()

    df = article_features.copy()
    topics = list(TOPIC_RULES.keys())

    def _row_directional_fields(row: pd.Series) -> pd.Series:
        bullish_conf = 0.0
        bearish_conf = 0.0
        active_topics: List[str] = []

        for topic in topics:
            topic_flag = int(row.get(f"topic_{topic}", 0) or 0)
            if topic_flag <= 0:
                continue

            active_topics.append(topic)
            confidence = float(row.get(f"{topic}_confidence", 0.0) or 0.0)
            strongest = row.get(f"{topic}_strongest_phrase")
            sign = topic_direction_sign(topic, strongest)
            if sign > 0:
                bullish_conf += confidence
            elif sign < 0:
                bearish_conf += confidence

        sentiment = bullish_conf - bearish_conf
        if sentiment > 0:
            direction_label = "bullish"
        elif sentiment < 0:
            direction_label = "bearish"
        else:
            direction_label = "neutral"

        return pd.Series(
            {
                "rule_based_topics": ",".join(active_topics),
                "bullish_confidence": round(bullish_conf, 6),
                "bearish_confidence": round(bearish_conf, 6),
                "rule_based_sentiment": round(sentiment, 6),
                "direction_label": direction_label,
            }
        )

    directional = df.apply(_row_directional_fields, axis=1)
    return pd.concat([df, directional], axis=1)


def explode_directional_topic_signals(
    article_features: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Convert wide article-level classifier outputs into one row per article-topic.

    The resulting long frame is suitable for generic weekly indicator logic.
    """
    if article_features.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                date_col,
                "category",
                "category_weight",
                "sentiment",
                "score",
                "confidence",
                "match_count",
                "surprise_count",
                "high_signal",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for _, row in article_features.iterrows():
        active_topics = [
            topic
            for topic in TOPIC_RULES
            if int(row.get(f"topic_{topic}", 0) or 0) > 0
        ]
        if not active_topics:
            continue

        category_weight = 1.0 / len(active_topics)
        for topic in active_topics:
            confidence = float(row.get(f"{topic}_confidence", 0.0) or 0.0)
            strongest = row.get(f"{topic}_strongest_phrase")
            sign = topic_direction_sign(topic, strongest)

            rows.append(
                {
                    "article_id": row.get("article_id"),
                    date_col: row.get(date_col),
                    "category": topic,
                    "category_weight": category_weight,
                    "sentiment": round(sign * confidence, 6),
                    "score": float(row.get(f"{topic}_score", 0.0) or 0.0),
                    "confidence": confidence,
                    "match_count": int(row.get(f"{topic}_match_count", 0) or 0),
                    "surprise_count": int(row.get(f"{topic}_surprise_count", 0) or 0),
                    "high_signal": int(row.get(f"{topic}_high_signal", 0) or 0),
                    "direction_sign": sign,
                }
            )

    return pd.DataFrame(rows)


def build_rule_based_weekly_indicators(
    article_features: pd.DataFrame,
    date_col: str = "date",
    decay_lambda: float = 0.1,
    decay_horizon_weeks: int = 4,
) -> pd.DataFrame:
    long_df = explode_directional_topic_signals(article_features, date_col=date_col)
    return compute_weekly_indicators_long(
        long_df,
        date_col=date_col,
        category_col="category",
        sentiment_col="sentiment",
        article_id_col="article_id",
        category_weight_col="category_weight",
        categories=list(TOPIC_RULES.keys()),
        decay_lambda=decay_lambda,
        decay_horizon_weeks=decay_horizon_weeks,
    )


def build_rule_based_indicator_column_names() -> List[str]:
    cols: List[str] = []
    for topic in TOPIC_RULES:
        for indicator in ["intensity", "sentiment", "decay", "variance"]:
            cols.append(f"{topic}_{indicator}")
    return cols


def build_rule_based_feature_column_names(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "date"]
