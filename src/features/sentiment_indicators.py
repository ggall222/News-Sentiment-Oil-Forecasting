# src/features/sentiment_indicators.py
#
# Computes 4 weekly sentiment indicators for each of the 8 topic categories.
#
# For each (week, category) pair:
#
#   1. category_intensity   (CI)  = (1 / N_t) * Σ_j CW_{i,j}
#      where CW_{i,j}=1/N_j for multi-label article j that contains category i.
#
#   2. sentiment_intensity  (CSI) = mean sentiment of category i articles in week t.
#
#   3. sentiment_decay      (CS_DI) = CSI_{i,t} + Σ_{l=1}^{t-1} exp(-(t-l)/n) * CSI_{i,l}
#      implemented with decay_horizon_weeks = n and decay_lambda = 1/n by default.
#
#   4. sentiment_variance   (CSI_V) = CI_{i,t} * Var(sentiment in category i, week t)
#
# Output: wide DataFrame with columns named  {category}_{indicator},
#         indexed by week-ending Friday date, aligned to the WTI price series.

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.features.topic_classifier import CATEGORIES

DECAY_LAMBDA = 0.1   # exponential decay rate; tune for your data


def _week_end_friday(date: pd.Timestamp) -> pd.Timestamp:
    """Return the Friday on or after `date`."""
    days_ahead = (4 - date.weekday()) % 7
    return date + pd.Timedelta(days=days_ahead)


def compute_weekly_indicators_long(
    df: pd.DataFrame,
    date_col: str = "date",
    category_col: str = "category",
    sentiment_col: str = "sentiment",
    article_id_col: str = "article_id",
    category_weight_col: str = "category_weight",
    categories: Optional[List[str]] = None,
    decay_lambda: float = DECAY_LAMBDA,
    decay_horizon_weeks: int = 4,
) -> pd.DataFrame:
    """
    Compute weekly indicators from a long article-category signal frame.

    The input must have one row per (article, category) pair.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, category_col])
    if df.empty:
        return pd.DataFrame(columns=["date"])

    if categories is None:
        categories = sorted(df[category_col].dropna().unique().tolist())
    else:
        categories = list(categories)

    if category_weight_col not in df.columns:
        df[category_weight_col] = 1.0
    if article_id_col not in df.columns:
        df[article_id_col] = np.arange(len(df))

    df["week_end"] = df[date_col].apply(_week_end_friday)
    weeks = sorted(df["week_end"].unique())
    records = []

    weekly_total_news = (
        df.groupby("week_end")[article_id_col]
        .nunique()
        .to_dict()
    )

    csi_history: Dict[str, Dict[pd.Timestamp, float]] = {c: {} for c in categories}

    if decay_lambda == DECAY_LAMBDA and decay_horizon_weeks > 0:
        decay_lambda = 1.0 / decay_horizon_weeks

    for week in weeks:
        row: Dict[str, object] = {"date": week}
        n_t = int(weekly_total_news.get(week, 0))

        for cat in categories:
            cat_articles = df[
                (df["week_end"] == week) &
                (df[category_col] == cat)
            ]

            n = len(cat_articles)
            scores = cat_articles[sentiment_col].dropna()

            if n_t > 0 and n > 0:
                ci_val = float(cat_articles[category_weight_col].sum() / n_t)
            else:
                ci_val = 0.0
            row[f"{cat}_intensity"] = round(ci_val, 6)

            if n == 0 or scores.empty:
                row[f"{cat}_sentiment"] = 0.0
                row[f"{cat}_decay"] = 0.0
                row[f"{cat}_variance"] = 0.0
                csi_history[cat][week] = 0.0
                continue

            csi_val = float(scores.mean())
            row[f"{cat}_sentiment"] = round(csi_val, 6)
            csi_history[cat][week] = csi_val

            var_val = float(scores.var(ddof=1)) if n > 1 else 0.0
            row[f"{cat}_variance"] = round(ci_val * var_val, 6)

        records.append(row)

    week_index = {w: i for i, w in enumerate(weeks)}
    for row in records:
        week = row["date"]
        t_idx = week_index[week]
        for cat in categories:
            csi_t = csi_history[cat].get(week, 0.0)
            decay_sum = csi_t
            for prev_week in weeks[:t_idx]:
                l_idx = week_index[prev_week]
                lag = t_idx - l_idx
                decay_sum += float(np.exp(-decay_lambda * lag) * csi_history[cat].get(prev_week, 0.0))
            row[f"{cat}_decay"] = round(decay_sum, 6)

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    return result


def compute_weekly_indicators(
    df: pd.DataFrame,
    date_col: str = "date",
    sentiment_col: str = "lm_sentiment",
    topics_col: str = "lm_topics",
    article_id_col: str = "article_id",
    categories: Optional[List[str]] = None,
    decay_lambda: float = DECAY_LAMBDA,
    decay_horizon_weeks: int = 4,
) -> pd.DataFrame:
    """
    Compute 4 × 8 = 32 weekly sentiment indicators.

    Parameters
    ----------
    df           : article-level DataFrame with date, lm_sentiment, lm_topics
    date_col     : name of the date column (datetime or date)
    sentiment_col: name of the LM-S sentiment score column
    topics_col   : name of the comma-separated topics column
    decay_lambda : decay rate λ for sentiment_decay indicator

    Returns
    -------
    DataFrame with one row per week-ending Friday and 32 indicator columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Explode topics so each article appears once per matched category
    df["topics_list"] = df[topics_col].apply(
        lambda x: x.split(",") if isinstance(x, str) and x else []
    )
    df["num_categories"] = df["topics_list"].apply(len)
    df["category_weight"] = df["num_categories"].apply(lambda n: 1.0 / n if n > 0 else 0.0)

    exploded = df.explode("topics_list").rename(columns={"topics_list": "category"})
    categories = list(categories or CATEGORIES)
    exploded = exploded[exploded["category"].isin(categories)].copy()

    return compute_weekly_indicators_long(
        exploded,
        date_col=date_col,
        category_col="category",
        sentiment_col=sentiment_col,
        article_id_col=article_id_col,
        category_weight_col="category_weight",
        categories=categories,
        decay_lambda=decay_lambda,
        decay_horizon_weeks=decay_horizon_weeks,
    )


def build_indicator_column_names() -> List[str]:
    """Return the full list of 32 indicator column names in order."""
    cols = []
    for cat in CATEGORIES:
        for indicator in ["intensity", "sentiment", "decay", "variance"]:
            cols.append(f"{cat}_{indicator}")
    return cols
