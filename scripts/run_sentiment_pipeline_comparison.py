"""
Compare FinBERT, LM-S, and rule-based weekly feature matrices under the same
walk-forward setup, with an optional combined-feature model.

Usage:
    python3 scripts/run_sentiment_pipeline_comparison.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.sentiment_pipeline_comparison import (
    DISPLAY_COLUMNS,
    run_sentiment_pipeline_comparison,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR = Path(__file__).parent.parent
FEATURE_DIR = BASE_DIR / "data" / "features"
FINBERT_PATH = FEATURE_DIR / "finbert_feature_matrix.parquet"
LM_PATH = FEATURE_DIR / "lm_feature_matrix.parquet"
RULE_BASED_PATH = FEATURE_DIR / "rule_based_topic_feature_matrix.parquet"


def parse_args():
    p = argparse.ArgumentParser(description="Compare weekly sentiment feature matrices")
    p.add_argument("--finbert-path", type=Path, default=FINBERT_PATH)
    p.add_argument("--lm-path", type=Path, default=LM_PATH)
    p.add_argument("--rule-based-path", type=Path, default=RULE_BASED_PATH)
    p.add_argument("--output-dir", type=Path, default=FEATURE_DIR)
    p.add_argument("--min-train-size", type=int, default=156)
    p.add_argument("--step", type=int, default=1)
    p.add_argument(
        "--include-combined",
        action="store_true",
        help="Also evaluate a model using all three feature sets concatenated.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    comparison, backtests, configs = run_sentiment_pipeline_comparison(
        finbert_path=args.finbert_path,
        lm_path=args.lm_path,
        rule_based_path=args.rule_based_path,
        output_dir=args.output_dir,
        min_train_size=args.min_train_size,
        step=args.step,
        include_combined=args.include_combined,
    )

    anchor = next(iter(configs.values()))
    logging.info(
        "Aligned comparison window: %d shared weekly rows (%s → %s)",
        len(anchor),
        anchor["date"].min().date(),
        anchor["date"].max().date(),
    )
    for name, df in configs.items():
        logging.info(
            "Completed walk-forward comparison for %s (%d rows, %d cols); saved %d OOS rows",
            name,
            len(df),
            len(df.columns),
            len(backtests[name]),
        )

    comparison_path = args.output_dir / "sentiment_pipeline_comparison.csv"

    print("\nSentiment pipeline comparison")
    print(comparison[DISPLAY_COLUMNS].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved comparison table → {comparison_path}")
