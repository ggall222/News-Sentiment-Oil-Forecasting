# Oil News Sentiment Analysis Test

This repository builds weekly WTI forecasting features from oil news, macro data, and price history.

It contains three parallel sentiment / topic feature branches plus shared evaluation code:

- `build_features.py`: standard FinBERT-based feature matrix
- `build_lm_sentiment_features.py`: LM-S lexicon-based feature matrix
- `build_rule_based_topic_features.py`: rule-based directional topic feature matrix
- `run_sentiment_pipeline_comparison.py`: apples-to-apples forecasting comparison across methods

The repo is structured so that a fresh clone can be run from the project root on any machine after installing dependencies and creating a `.env` file.

## Repository Layout

- `src/`: reusable data, feature engineering, and forecasting code
- `scripts/`: command-line entry points
- `data/raw/lexicons/`: committed lexicons and approved rule overrides
- `data/raw/benzinga/`: local raw Benzinga backfill parquet files
- `data/raw/oilprice/`: local raw OilPrice parquet files
- `data/features/`: local generated feature matrices and comparison outputs
- `notebooks/`: exploratory and comparison notebooks

## Python Setup

Use Python `3.9+`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` in the project root and fill in your keys:

```bash
cp .env.example .env
```

Required keys:

```bash
BENZINGA_API_KEY=your_benzinga_api_key
EIA_API_KEY=your_eia_api_key
FRED_API_KEY=your_fred_api_key
```

Used by:

- `BENZINGA_API_KEY`: Benzinga historical news backfill
- `EIA_API_KEY`: WTI price download and EIA macro series download
- `FRED_API_KEY`: FRED macro series download

## How To Run

Run all scripts from the repository root:

```bash
python3 scripts/<script_name>.py
```

The scripts resolve their input/output paths from the repo root, so they do not depend on your personal filesystem layout.

## Clean-Clone Run Order

If you are starting from a new clone with no local data:

```bash
python3 scripts/run_backfill.py
python3 scripts/oilprice_backfill.py
python3 scripts/fetch_wti_prices.py
python3 scripts/fetch_macro_features.py
python3 scripts/build_features.py
python3 scripts/build_finbert_sentiment_features.py
python3 scripts/build_lm_sentiment_features.py
python3 scripts/build_rule_based_topic_features.py
python3 scripts/run_sentiment_pipeline_comparison.py --include-combined
```

## What Each Step Produces

### Raw collection

```bash
python3 scripts/run_backfill.py
```

Creates month-partitioned Benzinga raw files under:

- `data/raw/benzinga/broad/YYYY-MM.parquet`
- `data/raw/benzinga/opec/YYYY-MM.parquet`
- `data/raw/benzinga/disruption/YYYY-MM.parquet`
- `data/raw/benzinga/macro/YYYY-MM.parquet`

```bash
python3 scripts/oilprice_backfill.py
```

or

```bash
python3 scripts/oilnewsscraper.py
```

Creates canonical OilPrice raw files under:

- `data/raw/oilprice/YYYY-MM.parquet`

### Price and macro inputs

```bash
python3 scripts/fetch_wti_prices.py
python3 scripts/fetch_macro_features.py
```

Outputs:

- `data/wti_prices.csv`
- `data/features/macro_features.parquet`

### Feature pipelines

```bash
python3 scripts/build_features.py
```

Outputs:

- `data/features/broad_daily.parquet`
- `data/features/opec_flags.parquet`
- `data/features/disruption_flags.parquet`
- `data/features/feature_matrix.parquet`

```bash
python3 scripts/build_finbert_sentiment_features.py
```

Outputs:

- `data/features/finbert_sentiment_weekly.parquet`
- `data/features/finbert_feature_matrix.parquet`

```bash
python3 scripts/build_lm_sentiment_features.py
```

Outputs:

- `data/features/lm_sentiment_weekly.parquet`
- `data/features/lm_feature_matrix.parquet`

```bash
python3 scripts/build_rule_based_topic_features.py
```

Outputs:

- `data/features/rule_based_topic_article_features.parquet`
- `data/features/rule_based_topic_weekly.parquet`
- `data/features/rule_based_topic_feature_matrix.parquet`

### Comparison

```bash
python3 scripts/run_sentiment_pipeline_comparison.py --include-combined
```

Outputs:

- `data/features/sentiment_pipeline_comparison.csv`
- `data/features/finbert_sentiment_comparison_backtest.parquet`
- `data/features/lm_sentiment_comparison_backtest.parquet`
- `data/features/rule_based_directional_comparison_backtest.parquet`
- `data/features/combined_all_comparison_backtest.parquet`

Metrics reported:

- `MAE`
- `Directional Accuracy`
- `DAER` (`directional_absolute_error_rate`)

## Optional Utilities

Expand keyword groups with Word2Vec and PMI filtering:

```bash
python3 scripts/expand_keywords.py
```

Mine rule-based directional phrase candidates:

```bash
python3 scripts/mine_rule_based_directional_phrases.py
python3 scripts/promote_rule_based_directional_phrases.py
```

Run the regime-aware backtest on the standard feature matrix:

```bash
python3 scripts/run_regime_backtest.py
```

## What Stays Local

These files are intentionally excluded from Git and should be regenerated locally after cloning:

- `.env`
- `data/raw/benzinga/`
- `data/raw/oilprice/`
- `data/features/`
- `data/wti_prices.csv`
- `data/expanded_keywords.json`
- `data/raw/oilprice/backfill_checkpoint.json`
- `data/raw/oilprice_relevant_articles.csv`

These are kept in Git because they affect reproducible behavior:

- `data/raw/lexicons/directional seed lexicon.csv`
- `data/raw/lexicons/rule_based_phrase_overrides.json`
- `data/raw/lexicons/rule_based_directional_phrase_review.csv`
- `data/raw/lexicons/seeds.py`

## Troubleshooting

### Missing API keys

If a script errors on missing keys, confirm `.env` exists in the repo root and contains the required variables.

### Missing raw data

If a build script cannot find Benzinga or OilPrice parquet inputs, run:

```bash
python3 scripts/run_backfill.py
python3 scripts/oilprice_backfill.py
```

### First FinBERT run is slow

The first FinBERT run may download `ProsusAI/finbert` from Hugging Face if it is not already cached locally.

### LM-S topic model fallback

The paragraph-topic pipeline can fall back to `LogisticRegression` if LightGBM runtime dependencies are unavailable. The pipeline still runs.
