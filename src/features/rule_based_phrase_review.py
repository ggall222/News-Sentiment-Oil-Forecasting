from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.features.rule_based_topic_classifier import TOPIC_RULES


PROMOTABLE_FIELDS = ("fixed_phrases", "direction_terms", "core_terms")
REVIEW_KEY_COLS = ["phrase", "label"]
REVIEW_DECISION_COLS = [
    "review_status",
    "approved_topic",
    "approved_field",
    "notes",
]
REVIEW_COLUMNS = [
    "phrase",
    "label",
    "suggested_topic",
    "suggested_field",
    "review_status",
    "approved_topic",
    "approved_field",
    "notes",
    "dominant_topics",
    "log_odds",
    "bullish_count",
    "bearish_count",
    "total_count",
    "bullish_share",
    "bearish_share",
    "in_latest_run",
]


def normalize_review_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"approved", "approve", "yes", "y", "true", "1"}:
        return "approved"
    if raw in {"rejected", "reject", "no", "n", "false", "0"}:
        return "rejected"
    if raw in {"hold", "held"}:
        return "hold"
    return "pending"


def _coerce_review_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["review_status"] = out["review_status"].apply(normalize_review_status)
    out["suggested_field"] = out["suggested_field"].replace("", "fixed_phrases")
    out["approved_field"] = out["approved_field"].replace("", "")
    out["in_latest_run"] = out["in_latest_run"].fillna(False).astype(bool)
    return out[REVIEW_COLUMNS]


def prepare_phrase_review(candidates: pd.DataFrame, review_path: Path | str) -> pd.DataFrame:
    """
    Refresh a review CSV while preserving prior approval decisions.
    """
    path = Path(review_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = _coerce_review_frame(pd.read_csv(path))
    else:
        existing = pd.DataFrame(columns=REVIEW_COLUMNS)

    existing_map = {
        (str(row["phrase"]), str(row["label"])): row
        for _, row in existing.iterrows()
    }

    current = candidates.copy()
    current["suggested_topic"] = current["dominant_topics"].fillna("").apply(
        lambda x: str(x).split(",")[0].strip() if str(x).strip() else ""
    )
    current["suggested_field"] = "fixed_phrases"
    current["review_status"] = "pending"
    current["approved_topic"] = ""
    current["approved_field"] = ""
    current["notes"] = ""
    current["in_latest_run"] = True

    merged_rows: List[dict] = []
    current_keys = set()
    for _, row in current.iterrows():
        key = (str(row["phrase"]), str(row["label"]))
        current_keys.add(key)
        merged = row.to_dict()
        prev = existing_map.get(key)
        if prev is not None:
            for col in REVIEW_DECISION_COLS:
                merged[col] = prev[col]
        if not merged.get("approved_topic"):
            merged["approved_topic"] = merged.get("suggested_topic", "")
        if not merged.get("approved_field"):
            merged["approved_field"] = merged.get("suggested_field", "fixed_phrases")
        merged_rows.append(merged)

    stale = existing[~existing.apply(lambda r: (str(r["phrase"]), str(r["label"])) in current_keys, axis=1)].copy()
    if not stale.empty:
        stale["in_latest_run"] = False
        merged_rows.extend(stale.to_dict(orient="records"))

    review_df = _coerce_review_frame(pd.DataFrame(merged_rows))
    review_df = review_df.sort_values(
        ["review_status", "in_latest_run", "total_count", "log_odds"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    review_df.to_csv(path, index=False)
    return review_df


def load_review_frame(review_path: Path | str) -> pd.DataFrame:
    path = Path(review_path)
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return _coerce_review_frame(pd.read_csv(path))


def build_phrase_overrides(review_df: pd.DataFrame) -> tuple[dict, list[str]]:
    """
    Build a runtime override payload from approved review rows.
    """
    overrides: Dict[str, Dict[str, List[str]]] = {}
    warnings: List[str] = []

    approved = review_df[review_df["review_status"] == "approved"].copy()
    for _, row in approved.iterrows():
        phrase = str(row.get("phrase", "")).strip().lower()
        topic = str(row.get("approved_topic") or row.get("suggested_topic") or "").strip()
        field = str(row.get("approved_field") or row.get("suggested_field") or "fixed_phrases").strip()

        if not phrase:
            warnings.append("Skipped empty approved phrase row.")
            continue
        if topic not in TOPIC_RULES:
            warnings.append(f"Skipped '{phrase}': unknown topic '{topic}'.")
            continue
        if field not in PROMOTABLE_FIELDS:
            warnings.append(f"Skipped '{phrase}': unsupported target field '{field}'.")
            continue

        overrides.setdefault(topic, {}).setdefault(field, [])
        if phrase not in overrides[topic][field]:
            overrides[topic][field].append(phrase)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approved_count": int(len(approved)),
        "topics": overrides,
    }
    return payload, warnings


def save_phrase_overrides(payload: dict, output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
