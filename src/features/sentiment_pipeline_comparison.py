from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.features.regime_aware_forecast import (
    backtest_metrics,
    walk_forward_backtest,
)


DISPLAY_COLUMNS = [
    "pipeline",
    "n_rows",
    "n_features",
    "n_oos",
    "mae",
    "directional_accuracy",
    "directional_absolute_error_rate",
    "oos_start",
    "oos_end",
]


def load_feature_matrix(path: Path | str, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} feature matrix not found: {path}")

    df = pd.read_parquet(path)
    required = {"date", "close", "close_pct_change"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} feature matrix is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def align_feature_sets(
    frames: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    common_dates = sorted(set.intersection(*(set(df["date"]) for df in frames.values())))
    if not common_dates:
        raise ValueError("No overlapping dates found between the requested feature matrices.")

    aligned: Dict[str, pd.DataFrame] = {}
    reference_label = next(iter(frames))
    reference = None
    for label, df in frames.items():
        cur = df[df["date"].isin(common_dates)].copy()
        cur = cur.sort_values("date").reset_index(drop=True)
        aligned[label] = cur

        if reference is None:
            reference = cur
            continue

        if not reference["date"].equals(cur["date"]):
            raise ValueError(f"Aligned matrices do not share the same date ordering: {reference_label} vs {label}")
        if not reference["close"].round(10).equals(cur["close"].round(10)):
            raise ValueError(f"Close prices differ on overlapping dates: {reference_label} vs {label}")
        if not reference["close_pct_change"].round(10).equals(cur["close_pct_change"].round(10)):
            raise ValueError(f"Targets differ on overlapping dates: {reference_label} vs {label}")

    non_feature_cols = {"date", "close", "close_pct_change"}
    combined_parts = []
    anchor = next(iter(aligned.values()))
    for label, df in aligned.items():
        combined_parts.append(df.drop(columns=list(non_feature_cols)).add_prefix(f"{label}__"))

    combined = pd.concat(
        [anchor[["date", "close", "close_pct_change"]], *combined_parts],
        axis=1,
    )
    return aligned, combined


def summarize_result(name: str, source_df: pd.DataFrame, backtest_df: pd.DataFrame) -> dict:
    metrics = backtest_metrics(backtest_df)
    feature_cols = [c for c in source_df.columns if c not in {"date", "close", "close_pct_change"}]

    return {
        "pipeline": name,
        "n_rows": int(len(source_df)),
        "n_features": int(len(feature_cols)),
        "n_oos": int(len(backtest_df)),
        "mae": float(metrics["mae"]),
        "directional_accuracy": float(metrics["directional_accuracy"]),
        "directional_absolute_error_rate": float(metrics["directional_absolute_error_rate"]),
        "oos_start": pd.to_datetime(backtest_df["date"]).min().date().isoformat(),
        "oos_end": pd.to_datetime(backtest_df["date"]).max().date().isoformat(),
    }


def run_sentiment_pipeline_comparison(
    finbert_path: Path | str,
    lm_path: Path | str,
    rule_based_path: Path | str,
    output_dir: Path | str | None = None,
    min_train_size: int = 156,
    step: int = 1,
    include_combined: bool = False,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    finbert_df = load_feature_matrix(finbert_path, "FinBERT")
    lm_df = load_feature_matrix(lm_path, "LM-S")
    rule_based_df = load_feature_matrix(rule_based_path, "Rule-based")

    aligned_frames, combined_df = align_feature_sets(
        {
            "finbert": finbert_df,
            "lm": lm_df,
            "rule_based": rule_based_df,
        }
    )

    configs = {
        "finbert_sentiment": aligned_frames["finbert"],
        "lm_sentiment": aligned_frames["lm"],
        "rule_based_directional": aligned_frames["rule_based"],
    }
    if include_combined:
        configs["combined_all"] = combined_df

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    backtests: Dict[str, pd.DataFrame] = {}
    for name, df in configs.items():
        bt = walk_forward_backtest(
            df,
            target_col="close_pct_change",
            date_col="date",
            min_train_size=min_train_size,
            step=step,
        )
        if bt.empty:
            raise RuntimeError(f"No out-of-sample rows produced for {name}.")

        backtests[name] = bt
        if out_dir is not None:
            bt.to_parquet(out_dir / f"{name}_comparison_backtest.parquet", index=False)
        results.append(summarize_result(name, df, bt))

    comparison = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    if out_dir is not None:
        comparison.to_csv(out_dir / "sentiment_pipeline_comparison.csv", index=False)

    return comparison, backtests, configs
