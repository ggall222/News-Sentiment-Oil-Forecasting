from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_oilprice_parquets(raw_dir: Path | str) -> pd.DataFrame:
    """
    Load canonical OilPrice parquet partitions from ``data/raw/oilprice``.

    Returns a de-duplicated DataFrame with ``date`` parsed to pandas datetime.
    Missing directories or empty partitions return an empty DataFrame.
    """
    path = Path(raw_dir)
    files = sorted(path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.empty:
        return df

    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    return df.reset_index(drop=True)
