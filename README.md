## Market Forecasting Service

This Flask application forecasts short-term prices for equities and leading crypto pairs, surfacing the context required to assess confidence in each prediction. In addition to historical pricing (via `yfinance`) and optional chart-image sentiment, the service now enriches every forecast with:

- Macro-economic indicators from FRED (GDP, CPI, PPI, Federal Funds cycle, synthetic 40-day Fed cadence, and derived inflation metrics)
- Company fundamentals (P/E ratios, PEG, dividend yield, beta, market cap, recent and upcoming earnings)
- News-driven sentiment buckets (overall, tariff, election, economic, global) with the latest headlines and per-article sentiment
- Price diagnostics (recent volatility, percentile bands, 1M / 1Q returns)

The front-end aggregates these signals so a user can scrutinise the forecast alongside the drivers used by the Prophet model.

---

### 1. Prerequisites

| Dependency | Purpose |
|------------|---------|
| Python 3.10+ | Runtime environment |
| `fredapi` | Access to FRED macro-economic time series |
| `vaderSentiment` | Lightweight news sentiment scoring |
| `yfinance`, `pandas`, `prophet` | Price history and forecasting |

Install Python requirements:

```
pip install -r requirements.txt
```

---

### 2. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | Recommended | Enables macro indicator fetches from FRED. Sign up for a free key at https://fred.stlouisfed.org. Without it, forecasts still run but macro regressors are disabled. |
| `NEWSAPI_KEY` | Optional | Augments company news with the NewsAPI Everything endpoint. Without it, the app falls back to Yahoo Finance headlines only. |
| `TOP_TECH_PAGE_PASSWORD` | Optional | Plaintext password that unlocks the Top Tech Stocks leaderboard (`tech-alpha-access` by default). |
| `TOP_TECH_PAGE_PASSWORD_HASH` | Optional | Werkzeug-compatible hash for the leaderboard password. Takes precedence over `TOP_TECH_PAGE_PASSWORD` when provided. |
| `TOP_TECH_TICKERS` | Optional | Comma-separated list of tickers to override the default US technology universe used in the leaderboard. |

Set variables locally (example):

```
export FRED_API_KEY="your_fred_api_key"
export NEWSAPI_KEY="your_newsapi_key"
```

---

### 3. Running the App

```
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5001
```

Navigate to `http://localhost:5001` and request a forecast:

1. Enter the ticker (e.g. `AAPL` or `BTC-USD`)
2. Choose the asset type. For crypto, the holiday selector auto-locks to a 24/7 calendar.
3. Choose market/holiday calendar (stocks only)
3. (Optional) Upload a chart image for additional sentiment weighting
4. Submit and review the forecast + enriched context (macro snapshot, fundamentals, news sentiment, recent price diagnostics, model regressors)

---

### 4. Top Tech Stock Leaderboard

Unlock the password-protected leaderboard at [`/top-tech`](http://localhost:5001/top-tech) to surface the strongest U.S. technology stocks for the upcoming sessions. The composite score blends:

- 14-day news sentiment and headline velocity (VADER on Yahoo Finance, optionally NewsAPI)
- Price momentum across 5-day, 1-month, and 3-month horizons plus realised volatility
- Point-in-time valuation and risk signals (market cap, forward P/E, PEG, beta)

Set a shared passcode through `TOP_TECH_PAGE_PASSWORD` (or its hashed companion `TOP_TECH_PAGE_PASSWORD_HASH`). Override the default tech universe via `TOP_TECH_TICKERS` when you need to include or exclude specific tickers.

The frontend calls `/api/top-tech-stocks` when you press **Generate Recommendations**, returning the ranked list, factor-level contributions, and recent news drivers for each pick.

---

### 5. Enriched Feature Engineering

`data_enrichment.py` orchestrates add-on data sources:

- **EconomicIndicatorFetcher** &mdash; pulls FRED series, derives inflation/ppi trends, and injects a 40-day Fed meeting cycle (`sin`, `cos`) for Prophet.
- **NewsSentimentFetcher** &mdash; aggregates Yahoo Finance (and optionally NewsAPI) headlines, categorises them by tariff/election/economic/global keywords, and summarises 7-day sentiment.
- **CompanyFundamentalsFetcher** &mdash; surfaces current valuation multiples and recent earnings cadence from Yahoo Finance fundamentals.

All regressors are standardised before being fed into Prophet. Future dates inherit the latest available macro values; news sentiment defaults to neutral beyond the last headline to avoid leaking stale tone into the forecast horizon.

---

### 6. Crypto Forecasting Notes

- Select the `Crypto` asset type (or the `Crypto (Global 24/7)` market) to ensure the forecast runs without weekend or holiday filters.
- Crypto dashboards and forecasts use the same Prophet workflow. Fundamentals may be sparse—Yahoo Finance does not expose valuation ratios for most digital assets.
- Macro and news enrichment continues to run; when specialised crypto macro indicators are needed, extend `data_enrichment.py` with alternative data sources.

### 7. Limitations & Next Steps

- Without a FRED key the macro-regressor benefits are disabled (flagged in the UI); consider enforcing the key or caching recent pulls.
- News sentiment currently relies on headline text; richer NLP (entity-level sentiment, LLM summarisation) could reduce noise.
- Prophet extrapolates macro features via forward-fill. For extended horizons consider separate forecasting of macro series.
- Image sentiment is still a rule-of-thumb placeholder; replace with a proper CNN if chart imagery remains a critical signal.

---

For deployment (Heroku, etc.), ensure environment variables are configured and the new dependencies are installed. The existing `Procfile` and `runtime.txt` continue to work unchanged.
