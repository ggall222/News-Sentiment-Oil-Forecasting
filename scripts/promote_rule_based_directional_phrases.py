# scripts/promote_rule_based_directional_phrases.py
#
# Promote reviewed directional phrase candidates into runtime rule overrides
# for the rule-based topic classifier.
#
# Workflow:
#   1. Run scripts/mine_rule_based_directional_phrases.py
#   2. Edit data/raw/lexicons/rule_based_directional_phrase_review.csv
#      and mark rows as approved/rejected/hold
#   3. Run this script to write approved phrase overrides
#
# Usage:
#   python3 scripts/promote_rule_based_directional_phrases.py

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.rule_based_phrase_review import (
    build_phrase_overrides,
    load_review_frame,
    save_phrase_overrides,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR = Path(__file__).parent.parent
REVIEW_CSV = BASE_DIR / "data/raw/lexicons/rule_based_directional_phrase_review.csv"
OVERRIDES_JSON = BASE_DIR / "data/raw/lexicons/rule_based_phrase_overrides.json"


if __name__ == "__main__":
    review_df = load_review_frame(REVIEW_CSV)
    if review_df.empty:
        logging.error("Review CSV not found or empty: %s", REVIEW_CSV)
        sys.exit(1)

    payload, warnings = build_phrase_overrides(review_df)
    save_phrase_overrides(payload, OVERRIDES_JSON)

    if warnings:
        for msg in warnings:
            logging.warning(msg)

    topics = payload.get("topics", {})
    approved_count = int(payload.get("approved_count", 0))
    logging.info("Saved approved rule overrides → %s", OVERRIDES_JSON)
    logging.info("Approved phrases promoted: %s", f"{approved_count:,}")
    for topic, field_map in sorted(topics.items()):
        total = sum(len(v) for v in field_map.values())
        logging.info("  %s: %s phrases", topic, f"{total:,}")

    print("\nRule-based phrase promotion complete:")
    print(f"  Review source : data/raw/lexicons/rule_based_directional_phrase_review.csv")
    print(f"  Output        : data/raw/lexicons/rule_based_phrase_overrides.json")
    print("  Effect        : future rule-based classifier runs will load these overrides automatically")
