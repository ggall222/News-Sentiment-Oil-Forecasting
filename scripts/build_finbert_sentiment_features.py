# scripts/build_finbert_sentiment_features.py
#
# Builds FinBERT-based weekly sentiment indicators using the same article
# corpus, topic assignment, weekly aggregation, and lag/merge flow as the
# LM-S branch.
#
# Usage:
#   python3 scripts/build_finbert_sentiment_features.py

import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.news_corpus import load_combined_broad_article_corpus
from src.features.news_features import score_sentiment
from src.features.paragraph_topic_classifier import assign_topics_from_paragraph_model
from src.features.sentiment_indicators import (
    build_indicator_column_names,
    compute_weekly_indicators,
)
from src.features.weekly_feature_pipeline import merge_weekly_features_with_price


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data/raw/benzinga"
OILPRICE_DIR = BASE_DIR / "data/raw/oilprice"
OUTPUT_DIR = BASE_DIR / "data/features"
PRICE_CSV = BASE_DIR / "data/wti_prices.csv"

DECAY_LAMBDA = 0.1
DECAY_HORIZON_WEEKS = 4


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading article data...")
    articles = load_combined_broad_article_corpus(RAW_DIR, OILPRICE_DIR)
    if articles.empty:
        logging.error("No article data found. Exiting.")
        sys.exit(1)

    source_counts = articles["source"].value_counts().to_dict()
    logging.info(f"Benzinga broad: {source_counts.get('benzinga', 0):,} articles")
    logging.info(f"OilPrice: {source_counts.get('oilprice', 0):,} articles")
    logging.info(f"Combined corpus: {len(articles):,} articles")

    logging.info("Scoring articles with FinBERT...")
    finbert_input = pd.DataFrame(
        {
            "id": articles["article_id"],
            "title": articles["headline"].fillna(""),
            "teaser": "",
            "body": articles["body_text"].fillna(""),
        }
    )
    finbert_scored = score_sentiment(
        finbert_input,
        title_col="title",
        teaser_col="teaser",
        body_col="body",
    )
    articles = articles.copy()
    articles["finbert_sentiment"] = finbert_scored["sentiment"].values
    logging.info(
        "  Mean score: %.4f  Std: %.4f",
        float(articles["finbert_sentiment"].mean()),
        float(articles["finbert_sentiment"].std()),
    )

    logging.info("Assigning oil-news topics with paragraph classifier...")
    articles, clf_info = assign_topics_from_paragraph_model(
        articles,
        text_col="text",
        date_col="date",
        topics_col="finbert_topics",
        paragraph_proba_threshold=0.40,
        min_chars=60,
    )
    logging.info(
        "  Paragraph classifier model=%s, paragraphs=%s, weak_labeled=%s",
        clf_info.get("model"), clf_info.get("paragraphs"), clf_info.get("labeled_paragraphs"),
    )
    covered = int((articles["finbert_topics"] != "").sum())
    logging.info(
        "  Articles with at least one topic: %s (%0.1f%%)",
        f"{covered:,}",
        100 * covered / len(articles),
    )

    for cat in build_indicator_column_names()[::4]:
        cat_name = cat.replace("_intensity", "")
        n = articles["finbert_topics"].str.contains(cat_name, na=False).sum()
        logging.info("    %s: %s articles", cat_name, f"{n:,}")

    logging.info("Computing weekly FinBERT indicators (4 × 8 = 32 features)...")
    weekly = compute_weekly_indicators(
        articles,
        date_col="date",
        sentiment_col="finbert_sentiment",
        topics_col="finbert_topics",
        article_id_col="article_id",
        decay_lambda=DECAY_LAMBDA,
        decay_horizon_weeks=DECAY_HORIZON_WEEKS,
    )
    weekly.to_parquet(OUTPUT_DIR / "finbert_sentiment_weekly.parquet", index=False)
    logging.info("  Saved → data/features/finbert_sentiment_weekly.parquet")

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
        merged.to_parquet(OUTPUT_DIR / "finbert_feature_matrix.parquet", index=False)
        logging.info("  Saved %d rows → data/features/finbert_feature_matrix.parquet", len(merged))

        covered_weeks = int((merged[lagged_cols[0]] != 0).sum()) if lagged_cols else 0
        logging.info(
            "  Weeks with news coverage: %d/%d (%0.1f%%)",
            covered_weeks,
            len(merged),
            100 * covered_weeks / len(merged) if len(merged) else 0.0,
        )
    else:
        logging.warning(f"Price CSV not found: {PRICE_CSV} — skipping merge")

    print("\nFinBERT weekly sentiment build complete:")
    print(f"  Articles processed   : {len(articles):,}")
    print(f"  Weekly indicator rows: {len(weekly):,}")
    print("  Outputs              :")
    print("    data/features/finbert_sentiment_weekly.parquet")
    if PRICE_CSV.exists():
        print("    data/features/finbert_feature_matrix.parquet")
