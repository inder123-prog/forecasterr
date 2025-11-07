import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from fredapi import Fred  # type: ignore
except ImportError:  # pragma: no cover - safety net if dependency missing
    Fred = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class NewsHighlights:
    all_headlines: List[Dict]
    sentiment_summary: Dict[str, float]


class EconomicIndicatorFetcher:
    """Fetch macro-economic indicators from FRED and shape them for time-series regressors."""

    SERIES_MAP = {
        "gdp": "GDP",
        "real_gdp": "GDPC1",
        "cpi": "CPIAUCSL",
        "core_cpi": "CPILFESL",
        "ppi": "PPIACO",
        "federal_funds": "DFF",
        "ppi_finished_goods": "WPSFD49207",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self._fred: Optional[Fred] = None

        if Fred is None:
            logger.warning("fredapi package is not installed; macro indicators will be unavailable.")
            return

        if not self.api_key:
            logger.warning(
                "FRED_API_KEY not provided. Set it in the environment to enable macro indicator enrichment."
            )
            return

        try:
            self._fred = Fred(api_key=self.api_key)
        except Exception as exc:
            logger.warning("Could not initialize FRED client: %s", exc)
            self._fred = None

    def _fetch_series(
        self, series_id: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.Series]:
        if self._fred is None:
            return None

        try:
            series = self._fred.get_series(series_id, start_date, end_date)
            if series is None or series.empty:
                logger.info("No data returned for FRED series %s", series_id)
                return None
            series.index = pd.to_datetime(series.index)
            return series.sort_index()
        except Exception as exc:  # pragma: no cover - external service
            logger.warning("Error fetching FRED series %s: %s", series_id, exc)
            return None

    def get_macro_dataframe(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Return a daily DataFrame of macro indicators covering the given range."""
        if self._fred is None:
            return pd.DataFrame(columns=["ds"])

        daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
        macro_df = pd.DataFrame({"ds": daily_index})

        for label, series_id in self.SERIES_MAP.items():
            series = self._fetch_series(series_id, start_date, end_date)
            if series is None or series.empty:
                continue

            daily_series = series.resample("D").ffill()
            daily_series = daily_series.reindex(daily_index, method="ffill")
            macro_df[label] = daily_series.values

        if macro_df.empty:
            return macro_df

        # Derived indicators
        macro_df["inflation_rate"] = (
            macro_df["cpi"].pct_change(periods=30) * 100
            if "cpi" in macro_df.columns
            else np.nan
        )
        macro_df["core_inflation_rate"] = (
            macro_df["core_cpi"].pct_change(periods=30) * 100
            if "core_cpi" in macro_df.columns
            else np.nan
        )
        macro_df["ppi_change_rate"] = (
            macro_df["ppi"].pct_change(periods=30) * 100
            if "ppi" in macro_df.columns
            else np.nan
        )

        if "federal_funds" in macro_df.columns:
            macro_df["federal_funds_delta"] = macro_df["federal_funds"].diff()
        else:
            macro_df["federal_funds_delta"] = np.nan

        # Synthetic 40-day federal cycle (captures FOMC cadence influence)
        days_since_start = (macro_df["ds"] - macro_df["ds"].min()).dt.days.astype(float)
        cycle_period = 40.0
        macro_df["fed_cycle_sin"] = np.sin(2 * np.pi * days_since_start / cycle_period)
        macro_df["fed_cycle_cos"] = np.cos(2 * np.pi * days_since_start / cycle_period)

        return macro_df

    @staticmethod
    def build_snapshot(macro_df: pd.DataFrame) -> Dict[str, float]:
        """Return the most recent values for quick reporting."""
        if macro_df.empty:
            return {}

        latest_row = macro_df.dropna(how="all").iloc[-1]
        snapshot_fields = [
            "gdp",
            "real_gdp",
            "cpi",
            "inflation_rate",
            "core_inflation_rate",
            "ppi",
            "ppi_change_rate",
            "federal_funds",
            "federal_funds_delta",
        ]
        snapshot = {}
        for field in snapshot_fields:
            if field in latest_row and pd.notna(latest_row[field]):
                snapshot[field] = float(latest_row[field])
        return snapshot


class NewsSentimentFetcher:
    """Aggregate latest company and macro news, compute sentiment buckets."""

    CATEGORY_KEYWORDS = {
        "tariff": [
            "tariff",
            "trade war",
            "import duty",
            "export tax",
            "sanction",
            "customs",
        ],
        "election": [
            "election",
            "ballot",
            "vote",
            "campaign",
            "poll",
            "primary",
        ],
        "economic": [
            "gdp",
            "inflation",
            "ppi",
            "cpi",
            "jobs report",
            "payrolls",
            "federal reserve",
            "fed",
            "central bank",
            "interest rate",
            "recession",
        ],
        "global": [
            "global",
            "geopolitical",
            "worldwide",
            "china",
            "europe",
            "asia",
            "uk",
            "brexit",
            "tariff",
            "sanction",
        ],
    }

    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY")
        self._sentiment = SentimentIntensityAnalyzer()
        self._session = requests.Session()

    def _pull_yfinance_news(self, ticker_symbol: str) -> List[Dict]:
        try:
            ticker = yf.Ticker(ticker_symbol)
            news_items = ticker.news or []
            return news_items[:50]  # limit to reduce noise
        except Exception as exc:
            logger.warning("Unable to fetch yfinance news for %s: %s", ticker_symbol, exc)
            return []

    def _pull_newsapi_news(
        self, ticker_symbol: str, start_date: datetime
    ) -> List[Dict]:  # pragma: no cover - network I/O
        if not self.newsapi_key:
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{ticker_symbol}" OR "{ticker_symbol} stock"',
            "from": start_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.newsapi_key,
        }

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            articles = payload.get("articles", [])
            normalized = []
            for article in articles:
                normalized.append(
                    {
                        "title": article.get("title"),
                        "summary": article.get("description") or "",
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name", "NewsAPI"),
                        "providerPublishTime": article.get("publishedAt"),
                    }
                )
            return normalized
        except Exception as exc:
            logger.warning("NewsAPI request failed: %s", exc)
            return []

    def _normalize_timestamp(self, raw_timestamp) -> Optional[pd.Timestamp]:
        if raw_timestamp is None:
            return None

        try:
            if isinstance(raw_timestamp, (int, float)):
                return pd.to_datetime(raw_timestamp, unit="s", utc=True).tz_convert(None)
            return pd.to_datetime(raw_timestamp, utc=True).tz_convert(None)
        except Exception:
            return None

    def _categorize(self, text: str) -> List[str]:
        categories = []
        lowered = text.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                categories.append(category)
        return categories

    def get_news_features(
        self,
        ticker_symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[pd.DataFrame, NewsHighlights]:
        news_records = []
        for item in self._pull_yfinance_news(ticker_symbol):
            timestamp = self._normalize_timestamp(item.get("providerPublishTime"))
            if timestamp is None or timestamp < start_date or timestamp > end_date:
                continue
            title = item.get("title") or ""
            summary = item.get("summary") or item.get("content") or ""
            categories = self._categorize(f"{title} {summary}")
            sentiment = self._sentiment.polarity_scores(title or summary)["compound"]
            news_records.append(
                {
                    "headline": title,
                    "summary": summary,
                    "url": item.get("link") or item.get("url"),
                    "source": item.get("provider") or item.get("source") or "yfinance",
                    "published": timestamp,
                    "sentiment": sentiment,
                    "categories": categories,
                }
            )

        # Optional NewsAPI enrichment
        if self.newsapi_key:
            for item in self._pull_newsapi_news(ticker_symbol, start_date):
                timestamp = self._normalize_timestamp(item.get("providerPublishTime"))
                if timestamp is None or timestamp < start_date or timestamp > end_date:
                    continue
                title = item.get("title") or ""
                summary = item.get("summary") or ""
                categories = self._categorize(f"{title} {summary}")
                sentiment = self._sentiment.polarity_scores(title or summary)["compound"]
                news_records.append(
                    {
                        "headline": title,
                        "summary": summary,
                        "url": item.get("url"),
                        "source": item.get("source") or "NewsAPI",
                        "published": timestamp,
                        "sentiment": sentiment,
                        "categories": categories,
                    }
                )

        if not news_records:
            empty_df = pd.DataFrame(columns=["ds", "news_sentiment"])
            highlights = NewsHighlights(all_headlines=[], sentiment_summary={})
            return empty_df, highlights

        news_df = pd.DataFrame(news_records)
        news_df["ds"] = news_df["published"].dt.normalize()

        category_columns = []
        for category in self.CATEGORY_KEYWORDS.keys():
            col_name = f"{category}_sentiment"
            category_columns.append(col_name)
            news_df[col_name] = news_df.apply(
                lambda row: row["sentiment"]
                if category in row["categories"]
                else np.nan,
                axis=1,
            )

        aggregated = (
            news_df.groupby("ds").agg(
                news_sentiment=("sentiment", "mean"),
                news_count=("sentiment", "count"),
                **{
                    f"{category}_sentiment": (f"{category}_sentiment", "mean")
                    for category in self.CATEGORY_KEYWORDS.keys()
                },
            )
        ).reset_index()

        # Fill NaNs for category sentiments with 0 (neutral) where unavailable
        category_sent_cols = [f"{category}_sentiment" for category in self.CATEGORY_KEYWORDS]
        aggregated[category_sent_cols] = aggregated[category_sent_cols].fillna(0.0)

        aggregated["news_sentiment"] = aggregated["news_sentiment"].fillna(0.0)

        highlights = NewsHighlights(
            all_headlines=news_df.sort_values("published", ascending=False)
            .head(10)[["headline", "published", "url", "source", "sentiment", "categories"]]
            .to_dict(orient="records"),
            sentiment_summary={
                "overall_sentiment": float(aggregated["news_sentiment"].tail(7).mean()),
                "overall_count": int(aggregated["news_count"].tail(7).sum()),
                **{
                    f"{category}_avg_sentiment": float(
                        aggregated[f"{category}_sentiment"].tail(7).mean()
                    )
                    for category in self.CATEGORY_KEYWORDS.keys()
                },
            },
        )

        aggregated.drop(columns=["news_count"], inplace=True)
        return aggregated, highlights


class CompanyFundamentalsFetcher:
    """Use yfinance to gather point-in-time company fundamentals."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def get_fundamentals(self, ticker_symbol: str) -> Dict:
        if ticker_symbol in self._cache:
            return self._cache[ticker_symbol]

        fundamentals: Dict[str, Optional[float]] = {}
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info or {}
            fundamentals.update(
                {
                    "trailing_pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield"),
                    "market_cap": info.get("marketCap"),
                    "beta": info.get("beta"),
                }
            )

            # Quarterly earnings
            try:
                earnings_df = ticker.quarterly_earnings
                if earnings_df is not None and not earnings_df.empty:
                    earnings_df = earnings_df.reset_index()
                    earnings_df.columns = ["quarter", "actual", "estimate"]
                    quarterly_data = earnings_df.tail(8).to_dict(orient="records")
                    fundamentals["quarterly_earnings"] = quarterly_data
            except Exception:
                fundamentals["quarterly_earnings"] = []

            # Upcoming earnings dates (limit 4)
            try:
                earnings_dates = ticker.get_earnings_dates(limit=4)
                if earnings_dates is not None and not earnings_dates.empty:
                    earnings_dates = earnings_dates.reset_index()
                    earnings_dates.columns = ["earnings_date", "eps_estimate", "eps_reported"]
                    calendar_records = earnings_dates.to_dict(orient="records")
                    for record in calendar_records:
                        earnings_dt = record.get("earnings_date")
                        if isinstance(earnings_dt, pd.Timestamp):
                            record["earnings_date"] = earnings_dt.isoformat()
                    fundamentals["earnings_calendar"] = calendar_records
            except Exception:
                fundamentals["earnings_calendar"] = []
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker_symbol, exc)

        # Ensure JSON-serializable primitives
        sanitized: Dict[str, Optional[float]] = {}
        for key, value in fundamentals.items():
            if isinstance(value, (np.integer, int)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value

        if isinstance(sanitized.get("quarterly_earnings"), list):
            for record in sanitized["quarterly_earnings"]:
                for sub_key, sub_value in list(record.items()):
                    if isinstance(sub_value, (np.floating, float)):
                        record[sub_key] = float(sub_value)
                    elif isinstance(sub_value, (np.integer, int)):
                        record[sub_key] = int(sub_value)
        if isinstance(sanitized.get("earnings_calendar"), list):
            for record in sanitized["earnings_calendar"]:
                for sub_key, sub_value in list(record.items()):
                    if isinstance(sub_value, (np.floating, float)):
                        record[sub_key] = float(sub_value)
                    elif isinstance(sub_value, (np.integer, int)):
                        record[sub_key] = int(sub_value)

        self._cache[ticker_symbol] = sanitized
        return sanitized


def _standardize_features(
    feature_df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if feature_df.empty or not feature_columns:
        return feature_df.copy(), {}

    stats: Dict[str, Dict[str, float]] = {}
    standardized = feature_df.copy()
    for col in feature_columns:
        if col not in standardized.columns:
            continue

        series = standardized[col].astype(float)
        mean = float(series.mean()) if not series.isna().all() else 0.0
        std = float(series.std()) if series.std() not in (0, np.nan) else 1.0
        std = std if std not in (0.0, np.nan) else 1.0
        standardized_col = f"{col}_z"
        standardized[standardized_col] = (series - mean) / std
        stats[col] = {"mean": mean, "std": std, "scaled_name": standardized_col}

    # Keep only scaled columns plus ds
    keep_cols = ["ds"] + [stat["scaled_name"] for stat in stats.values()]
    standardized = standardized[keep_cols]
    return standardized, stats


def build_enriched_dataset(
    ticker_symbol: str,
    price_df: pd.DataFrame,
    forecast_extension_days: int,
    market_country: str = "US",
) -> Dict:
    """Build enriched training data with macro indicators, news sentiment, and fundamentals."""

    if price_df.empty:
        raise ValueError("Price dataframe cannot be empty.")

    buffer_days = max(60, forecast_extension_days)
    start_date = (price_df["ds"].min() - pd.Timedelta(days=400)).to_pydatetime()
    end_date = (price_df["ds"].max() + pd.Timedelta(days=buffer_days)).to_pydatetime()

    econ_fetcher = EconomicIndicatorFetcher()
    macro_df = econ_fetcher.get_macro_dataframe(start_date, end_date)

    news_fetcher = NewsSentimentFetcher()
    news_df, news_highlights = news_fetcher.get_news_features(
        ticker_symbol, start_date, end_date
    )

    feature_master = pd.DataFrame({"ds": pd.date_range(start=start_date, end=end_date, freq="D")})
    if not macro_df.empty:
        feature_master = feature_master.merge(macro_df, on="ds", how="left")
    if not news_df.empty:
        feature_master = feature_master.merge(news_df, on="ds", how="left")

    # Fill macro indicators with forward/backward fill, news sentiment with zeros when missing
    macro_cols = [col for col in feature_master.columns if col not in {"ds"}]
    if macro_cols:
        for col in macro_cols:
            if col.startswith("news") or col.endswith("_sentiment"):
                feature_master[col] = feature_master[col].fillna(0.0)
            else:
                feature_master[col] = feature_master[col].ffill().bfill()

    feature_master, feature_stats = _standardize_features(
        feature_master, [col for col in feature_master.columns if col != "ds"]
    )

    # Merge with pricing data
    training_df = price_df.merge(feature_master, on="ds", how="left")
    training_df = training_df.sort_values("ds")
    training_df = training_df.fillna(0.0)

    # Build future regressors table
    future_regressors = feature_master.copy()

    fundamentals_fetcher = CompanyFundamentalsFetcher()
    fundamentals_snapshot = fundamentals_fetcher.get_fundamentals(ticker_symbol)

    price_metrics = _build_price_diagnostics(price_df)

    sanitized_news = []
    for item in news_highlights.all_headlines:
        sanitized_item = dict(item)
        published_value = sanitized_item.get("published")
        if isinstance(published_value, pd.Timestamp):
            sanitized_item["published"] = published_value.isoformat()
        sentiment_value = sanitized_item.get("sentiment")
        if isinstance(sentiment_value, (np.floating, float, int)):
            sanitized_item["sentiment"] = float(sentiment_value)
        sanitized_item["categories"] = list(sanitized_item.get("categories", []))
        sanitized_news.append(sanitized_item)

    sentiment_summary_serializable: Dict[str, Optional[float]] = {}
    for key, value in news_highlights.sentiment_summary.items():
        if isinstance(value, (np.integer, int)):
            sentiment_summary_serializable[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            sentiment_summary_serializable[key] = float(value)
        else:
            sentiment_summary_serializable[key] = value

    price_metrics_serialized: Dict[str, Optional[float]] = {}
    for key, value in price_metrics.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            price_metrics_serialized[key] = None
        elif isinstance(value, (np.floating, float)):
            price_metrics_serialized[key] = float(value)
        else:
            price_metrics_serialized[key] = value

    enriched_payload = {
        "training_df": training_df,
        "future_regressors": future_regressors,
        "regressor_columns": [
            col for col in training_df.columns if col not in {"ds", "y"}
        ],
        "news_highlights": sanitized_news,
        "news_sentiment_summary": sentiment_summary_serializable,
        "macro_snapshot": EconomicIndicatorFetcher.build_snapshot(macro_df),
        "fundamentals": fundamentals_snapshot,
        "feature_stats": feature_stats,
        "price_diagnostics": price_metrics_serialized,
        "has_macro_features": not macro_df.empty,
        "has_news_features": not news_df.empty,
    }
    return enriched_payload


def _build_price_diagnostics(price_df: pd.DataFrame) -> Dict[str, float]:
    diagnostics: Dict[str, float] = {}
    sorted_df = price_df.sort_values("ds")

    if sorted_df.empty:
        return diagnostics

    diagnostics["latest_close"] = float(sorted_df["y"].iloc[-1])
    diagnostics["fifth_percentile"] = float(sorted_df["y"].quantile(0.05))
    diagnostics["ninety_fifth_percentile"] = float(sorted_df["y"].quantile(0.95))

    # Rolling volatility and momentum
    sorted_df["returns"] = sorted_df["y"].pct_change()
    diagnostics["daily_volatility"] = float(sorted_df["returns"].std())
    diagnostics["monthly_return"] = float(
        (sorted_df["y"].iloc[-1] / sorted_df["y"].iloc[-22]) - 1
        if len(sorted_df) > 22
        else np.nan
    )
    diagnostics["quarter_return"] = float(
        (sorted_df["y"].iloc[-1] / sorted_df["y"].iloc[-66]) - 1
        if len(sorted_df) > 66
        else np.nan
    )

    return diagnostics

