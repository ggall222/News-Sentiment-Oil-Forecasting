from __future__ import annotations

from glob import glob
from pathlib import Path

import pandas as pd

from src.data.oilprice import load_oilprice_parquets


def load_benzinga_broad_articles(raw_dir: Path | str) -> pd.DataFrame:
    """
    Load Benzinga broad-tier articles into a canonical article corpus schema.
    """
    raw_path = Path(raw_dir)
    files = sorted(glob(str(raw_path / "broad" / "*.parquet")))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.empty:
        return df

    df = df.drop_duplicates(subset="id").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["headline"] = df["title"].fillna("")
    df["body_text"] = df["body"].fillna("")
    df["text"] = (df["headline"] + " " + df["body_text"]).str.strip()
    df["article_id"] = "benzinga::" + df["id"].astype(str)
    df["source"] = "benzinga"

    keep = ["article_id", "date", "headline", "body_text", "text", "source"]
    return df[keep].reset_index(drop=True)


def load_oilprice_articles(raw_dir: Path | str) -> pd.DataFrame:
    """
    Load canonical OilPrice parquet articles into the same corpus schema.
    """
    df = load_oilprice_parquets(raw_dir)
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["headline"] = df["title"].fillna(df["teaser"]).fillna("")
    df["body_text"] = df["body"].fillna("")
    df["text"] = (df["headline"] + " " + df["body_text"]).str.strip()
    df["article_id"] = "oilprice::" + df["id"].astype(str)
    df["source"] = "oilprice"

    keep = ["article_id", "date", "headline", "body_text", "text", "source"]
    return df[keep].reset_index(drop=True)


def load_combined_broad_article_corpus(
    benzinga_raw_dir: Path | str,
    oilprice_raw_dir: Path | str,
    include_oilprice: bool = True,
) -> pd.DataFrame:
    """
    Load the combined broad news corpus used by the LM-S and rule-based
    topic pipelines.
    """
    parts = []

    benzinga = load_benzinga_broad_articles(benzinga_raw_dir)
    if not benzinga.empty:
        parts.append(benzinga)

    if include_oilprice:
        oilprice = load_oilprice_articles(oilprice_raw_dir)
        if not oilprice.empty:
            parts.append(oilprice)

    if not parts:
        return pd.DataFrame(columns=["article_id", "date", "headline", "body_text", "text", "source"])

    articles = pd.concat(parts, ignore_index=True)
    articles = articles.dropna(subset=["date", "text"])
    articles = articles[articles["text"].str.strip() != ""].reset_index(drop=True)
    return articles
