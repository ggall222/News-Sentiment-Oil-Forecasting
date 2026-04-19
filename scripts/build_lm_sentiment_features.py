# scripts/build_lm_sentiment_features.py
#
# Builds LM-S weekly sentiment features across 8 oil-market topic categories.
#
# Pipeline:
#   1. Load Benzinga broad tier + OilPrice.com articles
#   2. Combine into unified text column (title + body)
#   3. Label price direction from headlines (rise / fall / None)
#   4. Build LM-S lexicon (base LM + oil extensions + corpus expansion)
#   5. Score all articles with LM-S lexicon
#   6. Classify articles into 8 topic categories
#   7. Compute 4 × 8 = 32 weekly indicators
#   8. Lag by 1 week + merge onto WTI price series → feature matrix
#   9. Save to data/features/lm_sentiment_weekly.parquet
#
# Usage (from WTI_News_API/ directory):
#   python3 scripts/build_lm_sentiment_features.py

import os
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.news_corpus import load_combined_broad_article_corpus
from src.features.lm_sentiment import (
    label_price_direction,
    build_lm_s_lexicon,
    score_articles,
)
from src.features.paragraph_topic_classifier import assign_topics_from_paragraph_model
from src.features.sentiment_indicators import (
    compute_weekly_indicators,
    build_indicator_column_names,
)
from src.features.weekly_feature_pipeline import merge_weekly_features_with_price

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.parent
RAW_DIR        = BASE_DIR / "data/raw/benzinga"
OILPRICE_DIR   = BASE_DIR / "data/raw/oilprice"
OUTPUT_DIR     = BASE_DIR / "data/features"
PRICE_CSV      = BASE_DIR / "data/wti_prices.csv"

DECAY_LAMBDA   = 0.1   # sentiment decay rate (higher = faster decay)
DECAY_HORIZON_WEEKS = 4
MIN_FREQ       = 3     # min word occurrences for lexicon expansion
MIN_RATIO      = 1.5   # min frequency ratio to add an expansion word

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and combine sources
    logging.info("Loading article data...")
    articles = load_combined_broad_article_corpus(RAW_DIR, OILPRICE_DIR)
    if articles.empty:
        logging.error("No article data found. Exiting.")
        sys.exit(1)

    source_counts = articles["source"].value_counts().to_dict()
    logging.info(f"Benzinga broad: {source_counts.get('benzinga', 0):,} articles")
    logging.info(f"OilPrice: {source_counts.get('oilprice', 0):,} articles")
    logging.info(f"Combined corpus: {len(articles):,} articles")

    # 2. Label price direction from headlines
    logging.info("Labeling price direction from headlines...")
    articles["price_direction"] = articles["headline"].apply(label_price_direction)
    labeled_counts = articles["price_direction"].value_counts(dropna=False)
    logging.info(f"  Labels: {labeled_counts.to_dict()}")

    # 3. Build LM-S lexicon
    logging.info("Building LM-S lexicon...")
    lexicon = build_lm_s_lexicon(
        articles,
        text_col="text",
        label_col="price_direction",
        min_freq=MIN_FREQ,
        min_ratio=MIN_RATIO,
    )
    logging.info(f"  Lexicon size: {len(lexicon):,} terms "
                 f"(positive: {sum(1 for v in lexicon.values() if v > 0)}, "
                 f"negative: {sum(1 for v in lexicon.values() if v < 0)})")

    # 4. Score all articles
    logging.info("Scoring articles with LM-S lexicon...")
    articles = score_articles(articles, lexicon, text_col="text")
    logging.info(f"  Mean score: {articles['lm_sentiment'].mean():.4f}  "
                 f"Std: {articles['lm_sentiment'].std():.4f}")

    # 5. Paragraph-level ML classification into 8 topic categories
    logging.info("Classifying paragraph-level topics with ML model...")
    articles, clf_info = assign_topics_from_paragraph_model(
        articles,
        text_col="text",
        date_col="date",
        topics_col="lm_topics",
        paragraph_proba_threshold=0.40,
        min_chars=60,
    )
    logging.info(
        "  Paragraph classifier model=%s, paragraphs=%s, weak_labeled=%s",
        clf_info.get("model"), clf_info.get("paragraphs"), clf_info.get("labeled_paragraphs"),
    )
    articles_with_topic = (articles["lm_topics"] != "").sum()
    logging.info(f"  Articles with at least one topic: {articles_with_topic:,} "
                 f"({100 * articles_with_topic / len(articles):.1f}%)")

    # Category distribution
    for cat in build_indicator_column_names()[::4]:  # every 4th = intensity cols
        cat_name = cat.replace("_intensity", "")
        n = articles["lm_topics"].str.contains(cat_name, na=False).sum()
        logging.info(f"    {cat_name}: {n:,} articles")

    # 6. Compute weekly indicators
    logging.info("Computing weekly sentiment indicators (4 × 8 = 32 features)...")
    weekly = compute_weekly_indicators(
        articles,
        date_col="date",
        sentiment_col="lm_sentiment",
        topics_col="lm_topics",
        decay_lambda=DECAY_LAMBDA,
        decay_horizon_weeks=DECAY_HORIZON_WEEKS,
    )
    logging.info(f"  Weekly indicator shape: {weekly.shape}")

    # 7. Save standalone weekly indicators
    weekly.to_parquet(OUTPUT_DIR / "lm_sentiment_weekly.parquet", index=False)
    logging.info("  Saved → data/features/lm_sentiment_weekly.parquet")

    # 8. Optionally merge with WTI price series
    if PRICE_CSV.exists():
        logging.info(f"Merging with price series: {PRICE_CSV}")
        indicator_cols = build_indicator_column_names()
        merged, lagged_cols = merge_weekly_features_with_price(
            weekly,
            PRICE_CSV,
            feature_cols=indicator_cols,
            lag_weeks=1,
            fill_value=0.0,
        )

        merged.to_parquet(OUTPUT_DIR / "lm_feature_matrix.parquet", index=False)
        logging.info(f"  Saved {len(merged)} rows → data/features/lm_feature_matrix.parquet")

        covered = (merged[lagged_cols[0]] != 0).sum()
        logging.info(f"  Weeks with news coverage: {covered}/{len(merged)} "
                     f"({100 * covered / len(merged):.1f}%)")
    else:
        logging.warning(f"Price CSV not found: {PRICE_CSV} — skipping merge")

    print("\nLM-S sentiment build complete:")
    print(f"  Articles processed  : {len(articles):,}")
    print(f"  Lexicon terms       : {len(lexicon):,}")
    print(f"  Weekly indicator rows: {len(weekly):,}")
    print(f"  Output              : data/features/lm_sentiment_weekly.parquet")
