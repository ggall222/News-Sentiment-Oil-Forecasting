# scripts/mine_rule_based_directional_phrases.py
#
# Mine bullish vs bearish directional phrase candidates from the same
# rule-based article corpus used by the standalone rule-based topic pipeline.
#
# Usage:
#   python3 scripts/mine_rule_based_directional_phrases.py

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.news_corpus import load_combined_broad_article_corpus
from src.features.rule_based_phrase_mining import mine_directional_phrases
from src.features.rule_based_phrase_review import prepare_phrase_review
from src.features.rule_based_topic_classifier import (
    RuleBasedOilTopicClassifier,
    build_directional_article_features,
    build_parquet_safe_article_features,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data/raw/benzinga"
OILPRICE_DIR = BASE_DIR / "data/raw/oilprice"
OUTPUT_DIR = BASE_DIR / "data/features"
OUT_CSV = OUTPUT_DIR / "rule_based_directional_phrase_candidates.csv"
REVIEW_CSV = BASE_DIR / "data/raw/lexicons/rule_based_directional_phrase_review.csv"

MIN_TOPIC_SCORE = 0.75
MIN_DOC_FREQ = 6
TOP_N_PER_LABEL = 250


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading article corpus...")
    articles = load_combined_broad_article_corpus(RAW_DIR, OILPRICE_DIR)
    if articles.empty:
        logging.error("No article data found. Exiting.")
        sys.exit(1)

    logging.info("Running rule-based directional classification...")
    classifier = RuleBasedOilTopicClassifier(min_topic_score=MIN_TOPIC_SCORE)
    article_features = classifier.classify_articles(
        articles,
        headline_col="headline",
        body_col="body_text",
        date_col="date",
        article_id_col="article_id",
    )
    article_features = build_parquet_safe_article_features(article_features, date_col="published_at")
    article_features = build_directional_article_features(article_features)

    mining_frame = article_features.merge(
        articles[["article_id", "text", "source"]],
        on="article_id",
        how="left",
    )

    candidates = mine_directional_phrases(
        mining_frame,
        text_col="text",
        label_col="direction_label",
        topics_col="rule_based_topics",
        min_doc_freq=MIN_DOC_FREQ,
        top_n_per_label=TOP_N_PER_LABEL,
    )

    if candidates.empty:
        logging.warning("No phrase candidates were mined from the classified corpus.")
        sys.exit(0)

    candidates.to_csv(OUT_CSV, index=False)
    logging.info("Saved %s directional phrase candidates → %s", f"{len(candidates):,}", OUT_CSV)

    review_df = prepare_phrase_review(candidates, REVIEW_CSV)
    pending = int((review_df["review_status"] == "pending").sum())
    approved = int((review_df["review_status"] == "approved").sum())
    logging.info(
        "Updated phrase review sheet → %s (pending=%s, approved=%s)",
        REVIEW_CSV,
        f"{pending:,}",
        f"{approved:,}",
    )

    bullish = int((candidates["label"] == "bullish").sum())
    bearish = int((candidates["label"] == "bearish").sum())
    print("\nRule-based directional phrase mining complete:")
    print(f"  Bullish candidates : {bullish:,}")
    print(f"  Bearish candidates : {bearish:,}")
    print("  Output             : data/features/rule_based_directional_phrase_candidates.csv")
    print("  Review sheet       : data/raw/lexicons/rule_based_directional_phrase_review.csv")
    print("  Next step          : python3 scripts/promote_rule_based_directional_phrases.py")
