from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def load_weekly_price_series(price_csv: Path | str) -> pd.DataFrame:
    price_df = pd.read_csv(price_csv, parse_dates=["date"])
    price_df["date"] = price_df["date"].dt.date
    return price_df


def lag_weekly_feature_frame(
    weekly_df: pd.DataFrame,
    feature_cols: Iterable[str],
    lag_weeks: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    df = weekly_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    lagged = df.copy()
    lagged["date"] = lagged["date"] + pd.Timedelta(weeks=lag_weeks)
    lagged["date"] = lagged["date"].dt.date

    cols = list(feature_cols)
    rename_map = {c: f"{c}_lag{lag_weeks}" for c in cols}
    lagged = lagged.rename(columns=rename_map)
    return lagged, list(rename_map.values())


def merge_weekly_features_with_price(
    weekly_df: pd.DataFrame,
    price_csv: Path | str,
    feature_cols: Iterable[str],
    lag_weeks: int = 1,
    fill_value: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    price_df = load_weekly_price_series(price_csv)
    lagged, lagged_cols = lag_weekly_feature_frame(weekly_df, feature_cols, lag_weeks=lag_weeks)

    merged = price_df.merge(lagged, on="date", how="left")
    merged[lagged_cols] = merged[lagged_cols].fillna(fill_value)
    merged["close_pct_change"] = merged["close"].pct_change() * 100
    merged = merged.dropna(subset=["close_pct_change"]).reset_index(drop=True)
    return merged, lagged_cols
