# scripts/build_rule_based_topic_features.py
#
# Builds standalone weekly directional topic features from the rule-based oil
# news topic classifier.
#
# Pipeline:
#   1. Load Benzinga broad tier + OilPrice.com articles
#   2. Run rule-based topic classification at the article level
#   3. Save standalone article-level topic features
#   4. Convert classifier output to LM-S-style directional article signals
#   5. Aggregate to week-ending Friday indicators
#   6. Lag by 1 week + merge onto WTI price series
#   7. Save standalone weekly and final feature matrices
#
# Usage (from project root):
#   python3 scripts/build_rule_based_topic_features.py

import os
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.news_corpus import load_combined_broad_article_corpus
from src.features.rule_based_topic_classifier import (
    RuleBasedOilTopicClassifier,
    build_parquet_safe_article_features,
    build_directional_article_features,
    build_rule_based_indicator_column_names,
    build_rule_based_weekly_indicators,
)
from src.features.weekly_feature_pipeline import merge_weekly_features_with_price


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR     = Path(__file__).parent.parent
RAW_DIR      = BASE_DIR / "data/raw/benzinga"
OILPRICE_DIR = BASE_DIR / "data/raw/oilprice"
OUTPUT_DIR   = BASE_DIR / "data/features"
PRICE_CSV    = BASE_DIR / "data/wti_prices.csv"

DECAY_LAMBDA    = 0.1
DECAY_HORIZON_WEEKS = 4
MIN_TOPIC_SCORE = 0.75


def log_topic_coverage(article_features: pd.DataFrame) -> None:
    topic_cols = sorted(c for c in article_features.columns if c.startswith("topic_"))
    covered = (article_features[topic_cols].sum(axis=1) > 0).sum() if topic_cols else 0
    logging.info(
        "  Articles with at least one rule-based topic: %s (%0.1f%%)",
        f"{covered:,}",
        100 * covered / len(article_features) if len(article_features) else 0.0,
    )

    for col in topic_cols:
        topic_name = col[len("topic_"):]
        n = int(article_features[col].sum())
        logging.info("    %s: %s articles", topic_name, f"{n:,}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the same broad article corpus used by the LM-S branch
    logging.info("Loading article data...")
    articles = load_combined_broad_article_corpus(RAW_DIR, OILPRICE_DIR)
    if articles.empty:
        logging.error("No article data found. Exiting.")
        sys.exit(1)

    source_counts = articles["source"].value_counts().to_dict()
    logging.info(f"Benzinga broad: {source_counts.get('benzinga', 0):,} articles")
    logging.info(f"OilPrice: {source_counts.get('oilprice', 0):,} articles")
    logging.info(f"Combined corpus: {len(articles):,} articles")

    # 2. Classify article-level topics
    logging.info("Running rule-based oil topic classification...")
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
    article_features = article_features.merge(
        articles[["article_id", "source"]],
        on="article_id",
        how="left",
    )

    article_out = OUTPUT_DIR / "rule_based_topic_article_features.parquet"
    article_features.to_parquet(article_out, index=False)
    logging.info(f"  Saved {len(article_features)} article rows → {article_out}")
    log_topic_coverage(article_features)

    # 3. Aggregate to weekly topic features
    logging.info("Aggregating weekly rule-based topic features...")
    weekly = build_rule_based_weekly_indicators(
        article_features,
        date_col="date",
        decay_lambda=DECAY_LAMBDA,
        decay_horizon_weeks=DECAY_HORIZON_WEEKS,
    )
    weekly_out = OUTPUT_DIR / "rule_based_topic_weekly.parquet"
    weekly.to_parquet(weekly_out, index=False)
    logging.info(f"  Saved {len(weekly)} weekly rows → {weekly_out}")

    # 4. Lag by 1 week + merge onto WTI prices
    if PRICE_CSV.exists():
        logging.info(f"Merging with price series: {PRICE_CSV}")
        weekly_feature_cols = build_rule_based_indicator_column_names()
        merged, lagged_cols = merge_weekly_features_with_price(
            weekly,
            PRICE_CSV,
            feature_cols=weekly_feature_cols,
            lag_weeks=1,
            fill_value=0.0,
        )

        matrix_out = OUTPUT_DIR / "rule_based_topic_feature_matrix.parquet"
        merged.to_parquet(matrix_out, index=False)
        logging.info(f"  Saved {len(merged)} rows → {matrix_out}")

        covered = (merged[lagged_cols].sum(axis=1) > 0).sum() if lagged_cols else 0
        logging.info(f"  Weeks with topic coverage: {covered}/{len(merged)} ({100 * covered / len(merged):.1f}%)")
    else:
        logging.warning(f"Price CSV not found: {PRICE_CSV} — skipping merge")

    print("\nRule-based topic feature build complete:")
    print(f"  Articles processed : {len(article_features):,}")
    print(f"  Weekly rows        : {len(weekly):,}")
    print("  Outputs            :")
    print("    data/features/rule_based_topic_article_features.parquet")
    print("    data/features/rule_based_topic_weekly.parquet")
    if PRICE_CSV.exists():
        print("    data/features/rule_based_topic_feature_matrix.parquet")
