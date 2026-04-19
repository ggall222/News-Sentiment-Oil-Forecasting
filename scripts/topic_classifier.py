from __future__ import annotations

import pandas as pd

from src.features.rule_based_topic_classifier import (
    TOPIC_RULES,
    TOPIC_DIRECTIONS,
    MatchRecord,
    RuleBasedOilTopicClassifier,
    build_parquet_safe_article_features,
    build_directional_article_features,
    build_rule_based_indicator_column_names,
    build_rule_based_feature_column_names,
    build_rule_based_weekly_indicators,
    build_weekly_topic_features,
    explode_directional_topic_signals,
)

__all__ = [
    "TOPIC_RULES",
    "TOPIC_DIRECTIONS",
    "MatchRecord",
    "RuleBasedOilTopicClassifier",
    "build_parquet_safe_article_features",
    "build_directional_article_features",
    "build_rule_based_indicator_column_names",
    "build_rule_based_feature_column_names",
    "build_rule_based_weekly_indicators",
    "build_weekly_topic_features",
    "explode_directional_topic_signals",
]


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "headline": [
                "Oil rises after larger-than-expected crude inventory draw and OPEC+ cuts",
                "WTI falls as OPEC considers boosting output and inventories rise",
                "Prices jump after pipeline attack disrupts exports in key region",
                "Crude slips as refinery restarts and demand weakens in Asia",
            ],
            "body_text": [
                "Traders reacted to a surprise draw in crude stocks and extended Saudi cuts.",
                "The market focused on record inventory build and possible quota increases.",
                "The disruption left barrels offline and tanker traffic at risk.",
                "Refinery operations resumed while weak fuel demand pressured the market.",
            ],
            "date": pd.to_datetime(
                ["2026-01-02", "2026-01-03", "2026-01-08", "2026-01-09"]
            ),
        }
    )

    clf = RuleBasedOilTopicClassifier()
    article_out = clf.classify_articles(
        demo,
        headline_col="headline",
        body_col="body_text",
        date_col="date",
    )
    article_out = build_parquet_safe_article_features(article_out, date_col="published_at")
    weekly_out = build_weekly_topic_features(article_out, date_col="date")

    print(article_out.filter(regex="^(article_id|top_label|topic_|.*_score$|.*_match_count$)").head())
    print(weekly_out.head())
