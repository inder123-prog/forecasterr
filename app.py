from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import date, timedelta, datetime # Added datetime for type checking
import logging
from PIL import Image
import io
import holidays as pyholidays # For public holidays
import os
import requests
import math
import json
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import secrets

from data_enrichment import build_enriched_dataset, CompanyFundamentalsFetcher, NewsSentimentFetcher
from ai_trading import generate_ai_trading_plan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FORECAST_APP_SECRET", "dev-secret-key")
CORS(app)

_DEFAULT_YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/"
}


def _build_yahoo_headers():
    headers = dict(_DEFAULT_YAHOO_HEADERS)
    override_ua = os.environ.get("YAHOO_FINANCE_USER_AGENT")
    if override_ua:
        headers["User-Agent"] = override_ua
    return headers


USER_STORE_PATH = Path(__file__).resolve().parent / 'users.json'

TOP_TECH_PASSWORD_SESSION_KEY = "tech_feature_authenticated"
_DEFAULT_TOP_TECH_PASSWORD = "tech-alpha-access"
_TOP_TECH_PASSWORD_HASH = os.environ.get("TOP_TECH_PAGE_PASSWORD_HASH")
_TOP_TECH_PASSWORD = os.environ.get("TOP_TECH_PAGE_PASSWORD", _DEFAULT_TOP_TECH_PASSWORD)

_DEFAULT_TECH_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "TSLA", "ORCL", "CRM",
    "ADBE", "CSCO", "AMD", "INTC", "QCOM", "TXN", "NOW", "AMAT", "LRCX", "MU",
    "PANW", "SNOW", "CRWD", "DDOG", "IBM", "SMCI", "ANET", "ADI", "NXPI", "KLAC"
]
_TECH_TICKERS_OVERRIDE = os.environ.get("TOP_TECH_TICKERS")
if _TECH_TICKERS_OVERRIDE:
    override_candidates = [
        ticker.strip().upper()
        for ticker in _TECH_TICKERS_OVERRIDE.split(",")
        if ticker.strip()
    ]
    _US_TECH_TICKERS = override_candidates or _DEFAULT_TECH_TICKERS
else:
    _US_TECH_TICKERS = _DEFAULT_TECH_TICKERS

_TOP_TECH_SCORING_FACTORS = [
    ("news_sentiment", 0.30, False),
    ("news_volume", 0.10, False),
    ("five_day_return", 0.18, False),
    ("one_month_return", 0.16, False),
    ("three_month_return", 0.10, False),
    ("market_cap_log", 0.08, False),
    ("forward_pe", 0.05, True),
    ("peg_ratio", 0.03, True),
    ("volatility", 0.08, True),
    ("beta", 0.02, True),
]


def _load_users():
    if not USER_STORE_PATH.exists():
        return {}
    try:
        with USER_STORE_PATH.open('r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse user store JSON: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error loading user store: {exc}", exc_info=True)
    return {}


def _save_users(users: dict):
    try:
        with USER_STORE_PATH.open('w', encoding='utf-8') as file:
            json.dump(users, file, indent=2)
    except Exception as exc:
        logger.error(f"Failed to persist user store: {exc}", exc_info=True)


def _require_authentication():
    return 'user' in session


def _verify_top_tech_password(candidate: str) -> bool:
    if not candidate:
        return False
    try:
        candidate_value = candidate.strip()
    except Exception:
        return False
    if _TOP_TECH_PASSWORD_HASH:
        try:
            return check_password_hash(_TOP_TECH_PASSWORD_HASH, candidate_value)
        except Exception:
            logger.warning("Invalid TOP_TECH_PAGE_PASSWORD_HASH configuration; falling back to plaintext comparison.")
    reference = _TOP_TECH_PASSWORD or _DEFAULT_TOP_TECH_PASSWORD
    try:
        return secrets.compare_digest(reference, candidate_value)
    except Exception:
        return reference == candidate_value


def _sanitize_news_highlights(items, limit: int = 3):
    sanitized = []
    if not items:
        return sanitized
    for item in items[:limit]:
        published = item.get("published")
        if isinstance(published, (pd.Timestamp, datetime)):
            try:
                published_iso = published.to_pydatetime().isoformat()
            except Exception:
                published_iso = published.isoformat()
        else:
            published_iso = published
        sentiment_value = item.get("sentiment")
        if isinstance(sentiment_value, (np.floating, float, int)):
            sentiment_value = float(sentiment_value)
        sanitized.append({
            "headline": item.get("headline"),
            "url": item.get("url"),
            "source": item.get("source"),
            "published": published_iso,
            "sentiment": sentiment_value,
            "categories": list(item.get("categories", []))
        })
    return sanitized


def _zscore_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    coerced = pd.to_numeric(series, errors='coerce')
    finite = coerced.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    mean = float(finite.mean())
    std = float(finite.std())
    if std in (0.0, np.nan) or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (coerced - mean) / std


def _safe_float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


# --- Tech stock ranking helper ---
def _compute_top_tech_rankings(limit: int = 10):
    candidates = list(dict.fromkeys(_US_TECH_TICKERS))
    if not candidates:
        return {
            "results": [],
            "universe_size": 0,
            "universe_tickers": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "weights": {}
        }

    fundamentals_fetcher = CompanyFundamentalsFetcher()
    news_fetcher = NewsSentimentFetcher()

    evaluation_records = []
    now_utc = datetime.utcnow()
    news_window_start = now_utc - timedelta(days=14)
    news_window_end = now_utc

    for raw_ticker in candidates:
        ticker_symbol = (raw_ticker or "").strip().upper()
        if not ticker_symbol:
            continue
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            price_hist = ticker_obj.history(period="6mo")
            if price_hist is None or price_hist.empty or "Close" not in price_hist.columns:
                logger.info(f"Skipping {ticker_symbol}: insufficient price history.")
                continue
            price_hist = price_hist.dropna(subset=["Close"])
            if price_hist.empty or len(price_hist) < 15:
                logger.info(f"Skipping {ticker_symbol}: not enough closing price observations.")
                continue

            closes = price_hist["Close"]
            latest_price = float(closes.iloc[-1])
            five_day_return = float(latest_price / closes.iloc[-6] - 1) if len(closes) > 6 else np.nan
            one_month_return = float(latest_price / closes.iloc[-22] - 1) if len(closes) > 22 else np.nan
            three_month_return = float(latest_price / closes.iloc[-66] - 1) if len(closes) > 66 else np.nan

            returns = closes.pct_change().dropna()
            volatility = float(returns.tail(60).std()) if not returns.empty else np.nan
            avg_volume = float(price_hist["Volume"].tail(21).mean()) if "Volume" in price_hist.columns else np.nan

            fundamentals = fundamentals_fetcher.get_fundamentals(ticker_symbol) or {}
            market_cap = fundamentals.get("market_cap")
            forward_pe = fundamentals.get("forward_pe")
            peg_ratio = fundamentals.get("peg_ratio")
            beta = fundamentals.get("beta")

            company_name = ticker_symbol
            try:
                info_payload = ticker_obj.get_info()
                if isinstance(info_payload, dict):
                    company_name = info_payload.get("shortName") or info_payload.get("longName") or ticker_symbol
            except Exception as info_error:
                logger.debug(f"Ticker info unavailable for {ticker_symbol}: {info_error}")

            news_df, news_highlights = news_fetcher.get_news_features(
                ticker_symbol, news_window_start, news_window_end
            )
            sentiment_summary = getattr(news_highlights, "sentiment_summary", {}) or {}
            news_sentiment = float(sentiment_summary.get("overall_sentiment") or 0.0)
            news_volume = float(sentiment_summary.get("overall_count") or 0)
            sanitized_highlights = _sanitize_news_highlights(
                getattr(news_highlights, "all_headlines", []), limit=3
            )

            evaluation_records.append({
                "ticker": ticker_symbol,
                "company_name": company_name,
                "latest_price": latest_price,
                "five_day_return": five_day_return,
                "one_month_return": one_month_return,
                "three_month_return": three_month_return,
                "volatility": volatility,
                "avg_volume": avg_volume,
                "market_cap": market_cap,
                "forward_pe": forward_pe,
                "peg_ratio": peg_ratio,
                "beta": beta,
                "news_sentiment": news_sentiment,
                "news_volume": news_volume,
                "news_highlights": sanitized_highlights
            })
        except Exception as exc:
            logger.warning(f"Failed to evaluate tech ticker {ticker_symbol}: {exc}", exc_info=True)
            continue

    if not evaluation_records:
        return {
            "results": [],
            "universe_size": 0,
            "universe_tickers": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "weights": {}
        }

    df = pd.DataFrame(evaluation_records)
    df["market_cap_log"] = np.log1p(
        pd.to_numeric(df.get("market_cap"), errors="coerce").fillna(0).clip(lower=0)
    )

    scoring_factors = list(_TOP_TECH_SCORING_FACTORS)

    component_series_lookup = {}
    composite_score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    for column, weight, invert in scoring_factors:
        if column not in df.columns:
            base_series = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
        else:
            base_series = _zscore_series(df[column])
        if invert:
            base_series = base_series * -1
        base_series = base_series.fillna(0.0)
        component = (base_series * weight).fillna(0.0)
        component_series_lookup[column] = component
        composite_score = composite_score.add(component, fill_value=0.0)

    df["composite_score"] = composite_score
    df_sorted = df.sort_values("composite_score", ascending=False).head(max(1, limit))

    results = []
    for rank, (idx, row) in enumerate(df_sorted.iterrows(), start=1):
        breakdown = {}
        for column, _, _ in scoring_factors:
            component_series = component_series_lookup.get(column)
            if component_series is None or component_series.empty:
                continue
            try:
                component_value = component_series.loc[idx]
                if component_value is None or not np.isfinite(component_value):
                    continue
                breakdown[column] = round(float(component_value), 4)
            except Exception:
                continue

        score_value = _safe_float(row.get("composite_score"))
        if score_value is not None:
            score_value = round(score_value, 4)

        results.append({
            "rank": rank,
            "ticker": row.get("ticker"),
            "companyName": row.get("company_name"),
            "score": score_value,
            "latestPrice": _safe_float(row.get("latest_price")),
            "marketCap": _safe_float(row.get("market_cap")),
            "forwardPE": _safe_float(row.get("forward_pe")),
            "pegRatio": _safe_float(row.get("peg_ratio")),
            "beta": _safe_float(row.get("beta")),
            "fiveDayReturn": _safe_float(row.get("five_day_return")),
            "oneMonthReturn": _safe_float(row.get("one_month_return")),
            "threeMonthReturn": _safe_float(row.get("three_month_return")),
            "volatility": _safe_float(row.get("volatility")),
            "averageVolume": _safe_float(row.get("avg_volume")),
            "newsSentiment": _safe_float(row.get("news_sentiment")),
            "newsVolume": int(row.get("news_volume")) if not pd.isna(row.get("news_volume")) else 0,
            "scoreBreakdown": breakdown,
            "newsHighlights": row.get("news_highlights") or []
        })

    weights_payload = {
        column: round(weight * (-1 if invert else 1), 3)
        for column, weight, invert in scoring_factors
    }

    return {
        "results": results,
        "universe_size": int(len(df)),
        "universe_tickers": list(df["ticker"]),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "weights": weights_payload
    }


# --- Helper function to get holidays for Prophet ---
def get_market_holidays(years, country='US'):
    normalized_country = (country or '').upper()
    if normalized_country in {"CRYPTO", "GLOBAL", "NONE", ""}:
        return pd.DataFrame(columns=['holiday', 'ds'])
    try:
        market_holidays_lib = pyholidays.CountryHoliday(normalized_country, years=years, prov=None, state=None)
    except Exception as exc:
        logger.warning(f"Failed to resolve holidays for country '{country}': {exc}")
        return pd.DataFrame(columns=['holiday', 'ds'])
    df_holidays = pd.DataFrame(columns=['holiday', 'ds'])
    if market_holidays_lib:
        holiday_dates = []
        holiday_names = []
        for dt, name in sorted(market_holidays_lib.items()):
            holiday_dates.append(dt)
            holiday_names.append(name)
        df_holidays['ds'] = pd.to_datetime(holiday_dates)
        df_holidays['holiday'] = holiday_names
        logger.info(f"Generated {len(df_holidays)} holidays for {country} for years: {years}")
    return df_holidays

# --- Image Analysis (Placeholder) ---
def analyze_chart_image(image_bytes):
    if not image_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        width, height = img.size
        if (width * height) % 100 < 33:
             logger.info("Mock Image Analysis: Trend UP")
             return "up"
        elif (width * height) % 100 < 66:
             logger.info("Mock Image Analysis: Trend NEUTRAL")
             return "neutral"
        else:
             logger.info("Mock Image Analysis: Trend DOWN")
             return "down"
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return None

# --- Stock Data Fetching ---
def get_stock_data(ticker_symbol, period="5y", include_ohlc=False):
    try:
        logger.info(f"Fetching data for ticker: {ticker_symbol}")
        stock = yf.Ticker(ticker_symbol)
        hist_data = stock.history(period=period)
        if hist_data.empty:
            logger.warning(f"No historical data found for {ticker_symbol} via yfinance history.")
            # Fallback: try to get some info if history is empty
            info = stock.info
            if info and info.get('regularMarketPrice') is not None and info.get('previousClose') is not None:
                 logger.warning(f"History empty for {ticker_symbol}, but info found. This might indicate issues.")
                 # This path means we can't train, but it's good to know.
            return None # Return None if no actual OHLCV history
        hist_data.reset_index(inplace=True)
        hist_data['ds'] = pd.to_datetime(hist_data['Date']).dt.tz_localize(None)
        hist_data['y'] = hist_data['Close']
        logger.info(f"Successfully fetched and processed data for {ticker_symbol}")
        price_df = hist_data[['ds', 'y']].copy()
        if include_ohlc:
            ohlc_columns = [col for col in ['ds', 'Open', 'High', 'Low', 'Close', 'Volume'] if col in hist_data.columns]
            ohlc_df = hist_data[ohlc_columns].copy()
            return price_df, ohlc_df
        return price_df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}", exc_info=True)
        return None

# --- Model Training and Prediction Logic ---
def train_and_forecast(
    data,
    days_to_forecast_daily,
    days_to_forecast_weekly,
    image_trend_regressor=None,
    country_holidays='US',
    regressor_columns=None,
    future_regressors=None
):
    if data is None or data.empty or len(data) < 2: # Prophet needs at least 2 data points
        logger.warning("Not enough data to train Prophet model.")
        return None, None, None

    min_hist_year = data['ds'].min().year
    max_forecast_date_approx = data['ds'].max() + pd.Timedelta(days=max(days_to_forecast_daily, days_to_forecast_weekly) + 60) # Increased buffer
    max_hist_year = max_forecast_date_approx.year
    holiday_years = list(range(min_hist_year, max_hist_year + 1))
    holidays_df = get_market_holidays(years=holiday_years, country=country_holidays)

    data_to_fit = data.copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        holidays=holidays_df if not holidays_df.empty else None
    )

    regressor_columns = regressor_columns or []

    if image_trend_regressor is not None and image_trend_regressor in ["up", "down", "neutral"]:
        if image_trend_regressor == "up": trend_value = 1
        elif image_trend_regressor == "down": trend_value = -1
        else: trend_value = 0
        data_to_fit['image_sentiment'] = trend_value
        if future_regressors is not None:
            future_regressors = future_regressors.copy()
            future_regressors['image_sentiment'] = trend_value
        if 'image_sentiment' not in regressor_columns:
            regressor_columns = regressor_columns + ['image_sentiment']
        logger.info(f"Prepared 'image_sentiment' regressor for fitting with value: {trend_value}")

    for reg_col in regressor_columns:
        if reg_col not in data_to_fit.columns:
            data_to_fit[reg_col] = 0.0
        model.add_regressor(reg_col)

    try:
        model.fit(data_to_fit)
    except Exception as e:
        logger.error(f"Error during model.fit: {e}", exc_info=True)
        return None, None, None

    future_daily_df = model.make_future_dataframe(periods=days_to_forecast_daily)
    future_weekly_df = model.make_future_dataframe(periods=days_to_forecast_weekly)

    if regressor_columns:
        if future_regressors is None or future_regressors.empty:
            logger.error("Future regressors are required when regressor columns are provided.")
            return None, None, None

        future_regressors_sorted = future_regressors.sort_values('ds')
        regressor_cols_with_ds = ['ds'] + regressor_columns

        future_daily_df = future_daily_df.merge(
            future_regressors_sorted[regressor_cols_with_ds],
            on='ds',
            how='left'
        )
        future_weekly_df = future_weekly_df.merge(
            future_regressors_sorted[regressor_cols_with_ds],
            on='ds',
            how='left'
        )

        for df in (future_daily_df, future_weekly_df):
            df.sort_values('ds', inplace=True)
            df[regressor_columns] = df[regressor_columns].ffill().bfill().fillna(0.0)

    elif 'image_sentiment' in data_to_fit.columns:
        future_trend_value = data_to_fit['image_sentiment'].iloc[-1]
        future_daily_df['image_sentiment'] = future_trend_value
        future_weekly_df['image_sentiment'] = future_trend_value

    try:
        forecast_daily = model.predict(future_daily_df)
        forecast_weekly = model.predict(future_weekly_df)
    except Exception as e:
        logger.error(f"Error during model.predict: {e}", exc_info=True)
        return None, None, None

    return model, forecast_daily, forecast_weekly

# --- Flask Routes ---
@app.route('/')
def index():
    if 'user' not in session:
        return render_template('login.html')
    return render_template('index.html', username=session.get('user'))


@app.route('/top-tech', methods=['GET', 'POST'])
def top_tech_portal():
    if not _require_authentication():
        return redirect(url_for('index'))

    error_message = None
    if request.method == 'POST':
        candidate_password = request.form.get('access_password', '')
        if _verify_top_tech_password(candidate_password):
            session[TOP_TECH_PASSWORD_SESSION_KEY] = True
            return redirect(url_for('top_tech_portal'))
        error_message = "Invalid access password. Please try again."

    access_granted = bool(session.get(TOP_TECH_PASSWORD_SESSION_KEY))
    scoring_weights = {
        factor[0]: round(factor[1] * (-1 if factor[2] else 1), 3)
        for factor in _TOP_TECH_SCORING_FACTORS
    }

    return render_template(
        'top_tech.html',
        username=session.get('user'),
        access_granted=access_granted,
        error_message=error_message,
        scoring_weights=scoring_weights
    )


@app.route('/api/register', methods=['POST'])
def register_user():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = (payload.get('password') or '').strip()
    email = (payload.get('email') or '').strip()

    if not username or not password or not email:
        return jsonify({"error": "Username, password, and email are required."}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters long."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    users = _load_users()
    if username in users:
        return jsonify({"error": "Username already exists. Please choose another."}), 400

    users[username] = {
        "password": generate_password_hash(password),
        "email": email
    }
    _save_users(users)
    session.pop(TOP_TECH_PASSWORD_SESSION_KEY, None)
    session['user'] = username
    return jsonify({"message": "Registration successful.", "username": username})


@app.route('/api/login', methods=['POST'])
def login_user():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = (payload.get('password') or '').strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    users = _load_users()
    user_record = users.get(username)
    if not user_record:
        return jsonify({"error": "Invalid username or password."}), 400

    if not check_password_hash(user_record.get('password', ''), password):
        return jsonify({"error": "Invalid username or password."}), 400

    session.pop(TOP_TECH_PASSWORD_SESSION_KEY, None)
    session['user'] = username
    return jsonify({"message": "Login successful.", "username": username})


@app.route('/api/logout', methods=['POST'])
def logout_user():
    session.pop('user', None)
    session.pop(TOP_TECH_PASSWORD_SESSION_KEY, None)
    return jsonify({"message": "Logged out."})


@app.route('/logout', methods=['GET'])
def logout_redirect():
    session.pop('user', None)
    session.pop(TOP_TECH_PASSWORD_SESSION_KEY, None)
    return redirect(url_for('index'))

@app.route('/forecast_with_image', methods=['POST'])
def forecast_stock_with_image():
    if not _require_authentication():
        return jsonify({"error": "Unauthorized"}), 401
    # Initialize variables to ensure they have default values
    image_trend = None
    image_sentiment_for_response = "Not Provided"
    stock_data_df = None
    current_price = None # Will be float or None
    last_date = None     # Will be datetime or None
    three_day_forecast_results = []
    week_ahead_pred_details = {
        "date": "N/A", "price": "N/A", 
        "lower_bound": "N/A", "upper_bound": "N/A", 
        "recommendation": "N/A"
    }
    ticker_symbol = "N/A" # Default if not found in form
    market_country = "US" # Default if not found
    asset_type = "equity"

    try:
        asset_type = (request.form.get('asset_type') or 'equity').strip().lower()
        ticker_symbol = request.form.get('ticker', 'N/A')  # Provide default
        image_file = request.files.get('chartImage')
        market_country = (request.form.get('market_country') or 'US')
        if isinstance(market_country, str):
            market_country = market_country.strip().upper() or 'US'
        else:
            market_country = 'US'

        enable_macro_features = (request.form.get('enable_macro', 'true').strip().lower() in {'true', '1', 'yes', 'on'})
        enable_news_features = (request.form.get('enable_news', 'true').strip().lower() in {'true', '1', 'yes', 'on'})
        enable_candlestick_features = (request.form.get('enable_candlestick', 'true').strip().lower() in {'true', '1', 'yes', 'on'})

        if not ticker_symbol or ticker_symbol == 'N/A':
            return jsonify({"error": "Ticker symbol is required"}), 400

        ticker_symbol = ticker_symbol.strip()
        valid_asset_types = {'equity', 'stock', 'etf', 'crypto'}
        if asset_type not in valid_asset_types:
            asset_type = 'equity'
        heuristic_crypto = ticker_symbol.upper().endswith(('-USD', '-USDT', '-BTC', '-ETH')) or market_country == 'CRYPTO'
        if asset_type != 'crypto' and heuristic_crypto:
            asset_type = 'crypto'
        is_crypto = asset_type == 'crypto'
        if is_crypto:
            market_country = 'CRYPTO'

        logger.info(
            f"Received forecast request for ticker: {ticker_symbol} with image: "
            f"{image_file.filename if image_file else 'No Image'}, Market: {market_country}, AssetType: {asset_type}"
        )

        if image_file and image_file.filename != '':
            try:
                image_bytes = image_file.read()
                analyzed_trend = analyze_chart_image(image_bytes)
                if analyzed_trend in ["up", "down", "neutral"]:
                    image_trend = analyzed_trend
                # image_trend remains None if analysis is inconclusive or returns unexpected
                image_sentiment_for_response = image_trend if image_trend else "Analysis Failed/Neutral"
                logger.info(f"Analyzed image trend result: {analyzed_trend}, Used image_trend: {image_trend}")
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}", exc_info=True)
                image_sentiment_for_response = "Error Processing Image"
                # image_trend remains None
        stock_data_response = get_stock_data(ticker_symbol, include_ohlc=True)
        if stock_data_response is None:
            logger.error(f"Failed to fetch historical data for {ticker_symbol}. Cannot proceed with forecast.")
            return jsonify({"error": f"Could not fetch historical data for {ticker_symbol}. It might be an invalid ticker or have no data."}), 404
        stock_data_df, stock_ohlc_df = stock_data_response
        if stock_data_df is None or stock_data_df.empty:
            logger.error(f"Failed to fetch historical data for {ticker_symbol}. Cannot proceed with forecast.")
            return jsonify({"error": f"Could not fetch historical data for {ticker_symbol}. It might be an invalid ticker or have no data."}), 404
        if stock_ohlc_df is None:
            stock_ohlc_df = pd.DataFrame()

        # We need to forecast enough periods to find 3 trading days and 1 week trading day
        days_for_daily_forecast_periods = 10
        days_for_week_ahead_periods = 15

        enrichment_payload = {
            "training_df": stock_data_df.copy(),
            "future_regressors": None,
            "regressor_columns": [],
            "news_highlights": [],
            "news_sentiment_summary": {},
            "macro_snapshot": {},
            "fundamentals": {},
            "price_diagnostics": {},
            "has_macro_features": False,
            "has_news_features": False,
            "candlestick_summary": {},
            "has_candlestick_features": False
        }
        training_input_df = stock_data_df.copy()
        regressor_columns = []
        future_regressors = None

        try:
            built_enrichment = build_enriched_dataset(
                ticker_symbol=ticker_symbol,
                price_df=stock_data_df.copy(),
                forecast_extension_days=max(days_for_daily_forecast_periods, days_for_week_ahead_periods) + 30,
                market_country=market_country,
                price_ohlc_df=stock_ohlc_df,
                enable_macro=enable_macro_features,
                enable_news=enable_news_features,
                enable_candlestick=enable_candlestick_features
            )
            enrichment_payload.update({
                key: built_enrichment.get(key, enrichment_payload.get(key))
                for key in enrichment_payload.keys()
                if key in built_enrichment
            })
            training_input_df = built_enrichment.get("training_df", training_input_df)
            regressor_columns = built_enrichment.get("regressor_columns", [])
            future_regressors = built_enrichment.get("future_regressors")
        except Exception as enrichment_error:
            logger.error(f"Failed to build enrichment payload for {ticker_symbol}: {enrichment_error}", exc_info=True)

        candlestick_summary = enrichment_payload.get("candlestick_summary", {}) or {}
        pattern_bias = str(candlestick_summary.get("bias", "neutral")).lower()
        pattern_buy_adjust = 0.0
        pattern_sell_adjust = 0.0
        if pattern_bias == "bullish":
            pattern_buy_adjust = -0.003
            pattern_sell_adjust = 0.002
        elif pattern_bias == "bearish":
            pattern_buy_adjust = 0.003
            pattern_sell_adjust = -0.002
        elif pattern_bias == "sideways":
            pattern_buy_adjust = 0.001
            pattern_sell_adjust = 0.001

        current_price = stock_data_df['y'].iloc[-1]
        last_date = stock_data_df['ds'].iloc[-1]

        model, forecast_output_daily_df, forecast_output_weekly_df = train_and_forecast(
            training_input_df.copy(),
            days_for_daily_forecast_periods,
            days_for_week_ahead_periods,
            image_trend_regressor=image_trend,
            country_holidays=market_country,
            regressor_columns=regressor_columns,
            future_regressors=future_regressors
        )

        if forecast_output_daily_df is None or forecast_output_weekly_df is None :
            logger.error(f"Model training or prediction failed for {ticker_symbol}.")
            return jsonify({"error": "Failed to generate forecast (model error)."}), 500
            
        min_hist_year_check = last_date.year # Start checking holidays from last known date's year
        max_forecast_date_approx_check = last_date + pd.Timedelta(days=max(days_for_daily_forecast_periods, days_for_week_ahead_periods) + 30)
        max_hist_year_check = max_forecast_date_approx_check.year
        custom_holidays_dates = []
        if not is_crypto:
            holiday_years_for_check = list(range(min_hist_year_check, max_hist_year_check + 1))
            custom_holidays_dates = get_market_holidays(years=holiday_years_for_check, country=market_country)['ds'].tolist()

        base_buy_threshold = 0.01
        base_sell_threshold = 0.01
        daily_buy_threshold = max(0.003, base_buy_threshold + pattern_buy_adjust)
        daily_sell_threshold = max(0.003, base_sell_threshold + pattern_sell_adjust)

        trading_days_found = 0
        future_calendar_dates_in_forecast = forecast_output_daily_df[forecast_output_daily_df['ds'] > last_date]['ds']

        for potential_trading_date in future_calendar_dates_in_forecast:
            if (not is_crypto) and potential_trading_date.dayofweek >= 5:
                continue
            if (not is_crypto) and any(potential_trading_date.normalize() == hol_date.normalize() for hol_date in custom_holidays_dates):
                continue

            prediction_row = forecast_output_daily_df[forecast_output_daily_df['ds'] == potential_trading_date].head(1)
            if not prediction_row.empty:
                price = round(prediction_row['yhat'].iloc[0], 2)
                lower = round(prediction_row['yhat_lower'].iloc[0], 2)
                upper = round(prediction_row['yhat_upper'].iloc[0], 2)
                recommendation = "Hold/Neutral"

                image_factor = 0.0
                if image_trend == "up":
                    image_factor = 0.002
                elif image_trend == "down":
                    image_factor = -0.002

                if isinstance(price, (int, float)) and isinstance(current_price, (int, float)) and current_price > 0:
                    upper_threshold = current_price * (1 + daily_buy_threshold + image_factor)
                    lower_threshold = current_price * (1 - daily_sell_threshold + image_factor)
                    if price > upper_threshold:
                        recommendation = "Consider Buying"
                    elif price < lower_threshold:
                        recommendation = "Consider Selling/Not Buying"

                bias_hint = candlestick_summary.get("bias")
                if isinstance(bias_hint, str) and bias_hint.lower() not in {"", "neutral"}:
                    recommendation = f"{recommendation} (candlestick bias: {bias_hint.capitalize()})"

                three_day_forecast_results.append({
                    "date": potential_trading_date.strftime('%Y-%m-%d'),
                    "price": price,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "recommendation": recommendation,
                    "candlestick_bias": candlestick_summary.get("bias"),
                    "candlestick_net_score": candlestick_summary.get("net_score")
                })
                trading_days_found += 1
                if trading_days_found >= 3:
                    break
        
        base_week_buy_threshold = 0.02
        base_week_sell_threshold = 0.02
        weekly_buy_threshold = max(0.01, base_week_buy_threshold + pattern_buy_adjust * 1.5)
        weekly_sell_threshold = max(0.01, base_week_sell_threshold + pattern_sell_adjust * 1.5)

        target_week_ahead_search_date = last_date + timedelta(days=7)
        for pot_wk_date in forecast_output_weekly_df[forecast_output_weekly_df['ds'] >= target_week_ahead_search_date]['ds']:
            if (not is_crypto) and pot_wk_date.dayofweek >= 5:
                continue
            if (not is_crypto) and any(pot_wk_date.normalize() == hol_date.normalize() for hol_date in custom_holidays_dates):
                continue
            
            wk_pred_row = forecast_output_weekly_df[forecast_output_weekly_df['ds'] == pot_wk_date].head(1)
            if not wk_pred_row.empty:
                wk_price = round(wk_pred_row['yhat'].iloc[0], 2)
                wk_rec = "Hold/Neutral"
                image_factor_wk = 0.0
                if image_trend == "up":
                    image_factor_wk = 0.004
                elif image_trend == "down":
                    image_factor_wk = -0.004

                if isinstance(wk_price, (int, float)) and isinstance(current_price, (int, float)) and current_price > 0:
                    upper_threshold_week = current_price * (1 + weekly_buy_threshold + image_factor_wk)
                    lower_threshold_week = current_price * (1 - weekly_sell_threshold + image_factor_wk)
                    if wk_price > upper_threshold_week:
                        wk_rec = "Consider Buying (Week Ahead)"
                    elif wk_price < lower_threshold_week:
                        wk_rec = "Consider Selling/Not Buying (Week Ahead)"

                bias_hint_week = candlestick_summary.get("bias")
                if isinstance(bias_hint_week, str) and bias_hint_week.lower() not in {"", "neutral"}:
                    wk_rec = f"{wk_rec} (candlestick bias: {bias_hint_week.capitalize()})"

                week_ahead_pred_details = {
                    "date": pot_wk_date.strftime('%Y-%m-%d'),
                    "price": wk_price,
                    "lower_bound": round(wk_pred_row['yhat_lower'].iloc[0], 2),
                    "upper_bound": round(wk_pred_row['yhat_upper'].iloc[0], 2),
                    "recommendation": wk_rec,
                    "candlestick_bias": candlestick_summary.get("bias"),
                    "candlestick_net_score": candlestick_summary.get("net_score")
                }
            break 
        
        current_price_display = f"{current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A"
        last_date_display = last_date.strftime('%Y-%m-%d') if isinstance(last_date, (date, pd.Timestamp, datetime)) else "undefined"
        week_ahead_price_display = week_ahead_pred_details.get('price', "N/A") # Use .get for safety

        try:
            dashboard_views = build_stock_dashboards(ticker_symbol)
        except Exception as dashboard_error:
            dashboard_views = {}
            logger.error(f"Failed to build dashboard snapshots for {ticker_symbol}: {dashboard_error}", exc_info=True)

        ai_trading_payload = {"status": "unavailable"}
        try:
            ai_trading_payload = generate_ai_trading_plan(
                ticker=ticker_symbol,
                ohlc_df=stock_ohlc_df,
                current_price=current_price
            )
        except Exception as ai_trading_error:
            ai_trading_payload = {
                "status": "error",
                "reason": "AI trading module failed to generate recommendations."
            }
            logger.error(
                f"Failed to build AI trading plan for {ticker_symbol}: {ai_trading_error}",
                exc_info=True
            )

        logger.info(
            f"Forecast for {ticker_symbol} (Image Trend: {str(image_trend)}, Market: {market_country}, "
            f"Candlestick Bias: {candlestick_summary.get('bias', 'neutral')}): "
            f"Current ${current_price_display} (as of {last_date_display}), "
            f"3-Day Forecast: {[(d['date'], d['price']) for d in three_day_forecast_results]}, "
            f"Week Ahead Prediction ({week_ahead_pred_details.get('date', 'N/A')}): ${week_ahead_price_display}"
        )

        regressor_columns_for_response = list(regressor_columns)
        if image_trend in ["up", "down", "neutral"] and 'image_sentiment' not in regressor_columns_for_response:
            regressor_columns_for_response.append('image_sentiment')

        response_payload = {
            "ticker": ticker_symbol,
            "current_price": current_price_display if current_price_display != "N/A" else None,  # Send None if N/A for JS
            "last_known_date": last_date_display,
            "image_analysis_sentiment": image_sentiment_for_response,
            "three_day_forecast": three_day_forecast_results,
            "forecast_week_ahead": week_ahead_pred_details,
            "price_diagnostics": enrichment_payload.get("price_diagnostics", {}),
            "macro_snapshot": enrichment_payload.get("macro_snapshot", {}),
            "news_sentiment_summary": enrichment_payload.get("news_sentiment_summary", {}),
            "news_highlights": enrichment_payload.get("news_highlights", []),
            "fundamentals": enrichment_payload.get("fundamentals", {}),
            "feature_flags": {
                "macro": enrichment_payload.get("has_macro_features", False),
                "news": enrichment_payload.get("has_news_features", False),
                "candlestick": enrichment_payload.get("has_candlestick_features", False),
                "fundamental": enrichment_payload.get("has_fundamental_features", False)
            },
            "feature_flags_requested": enrichment_payload.get("requested_feature_flags", {}),
            "candlestick_patterns": candlestick_summary,
            "candlestick_feature_columns": enrichment_payload.get("candlestick_feature_columns", []),
            "fundamental_feature_columns": enrichment_payload.get("fundamental_feature_columns", []),
            "fundamental_features_original": enrichment_payload.get("fundamental_features_original", []),
            "model_regressors": regressor_columns_for_response,
            "stock_dashboards": dashboard_views,
            "asset_type": "crypto" if is_crypto else "equity",
            "ai_trading": ai_trading_payload
        }

        feature_flag_notes = {}
        if enable_macro_features:
            if not enrichment_payload.get("has_macro_features", False):
                feature_flag_notes["macro"] = "Macro data unavailable for this forecast. Ensure FRED_API_KEY is configured and data exists for the requested window."
        else:
            feature_flag_notes["macro"] = "Macro data disabled for this request."
        if enable_news_features:
            if not enrichment_payload.get("has_news_features", False):
                feature_flag_notes["news"] = "No qualifying news sentiment was found for the requested window."
        else:
            feature_flag_notes["news"] = "News sentiment disabled for this request."
        if enable_candlestick_features:
            if not enrichment_payload.get("has_candlestick_features", False):
                feature_flag_notes["candlestick"] = "Candlestick patterns could not be derived (insufficient OHLC data)."
        else:
            feature_flag_notes["candlestick"] = "Candlestick analytics disabled for this request."
        if not enrichment_payload.get("has_fundamental_features", False):
            feature_flag_notes["fundamental"] = "Fundamental filings unavailable or incomplete for regression enrichment."

        response_payload["feature_flag_notes"] = feature_flag_notes

        return jsonify(response_payload)

    except Exception as e:
        # Log crucial variables at the point of error
        current_image_trend_on_error = "variable 'image_trend' not in scope"
        if 'image_trend' in locals(): current_image_trend_on_error = str(image_trend)
        
        current_ticker_on_error = "variable 'ticker_symbol' not in scope"
        if 'ticker_symbol' in locals(): current_ticker_on_error = str(ticker_symbol)

        logger.error(f"Error in /forecast_with_image endpoint for ticker '{current_ticker_on_error}' (image_trend state: {current_image_trend_on_error}): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please check logs."}), 500

def _prepare_history_frame(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()
    df = hist_df.copy().reset_index()
    date_column = None
    for candidate in ("Datetime", "Date", "index"):
        if candidate in df.columns:
            date_column = candidate
            break
    if date_column is None:
        return pd.DataFrame()
    df["ds"] = pd.to_datetime(df[date_column]).dt.tz_localize(None)
    df = df.sort_values("ds")
    return df

def _serialize_price_series(hist_df: pd.DataFrame, max_points: int = 120):
    if hist_df.empty or "ds" not in hist_df.columns or "Close" not in hist_df.columns:
        return []
    total_points = len(hist_df)
    if total_points <= max_points:
        sampled_df = hist_df
    else:
        step = max(1, math.ceil(total_points / max_points))
        indices = list(range(0, total_points, step))
        if indices[-1] != total_points - 1:
            indices.append(total_points - 1)
        sampled_df = hist_df.iloc[indices]
    series = []
    for _, row in sampled_df.iterrows():
        price = row.get("Close")
        timestamp = row.get("ds")
        if pd.isna(price) or pd.isna(timestamp):
            continue
        try:
            series.append({
                "time": timestamp.isoformat(),
                "price": round(float(price), 4)
            })
        except Exception:
            continue
    return series

def _compute_change(latest_value: float, reference_value: float):
    if reference_value in (None, 0) or pd.isna(reference_value):
        return None, None
    change = latest_value - reference_value
    percent = (change / reference_value) * 100 if reference_value != 0 else None
    return change, percent

def _safe_round(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None

def _safe_int(value):
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None

def build_stock_dashboards(ticker_symbol: str):
    dashboards = {}
    if not ticker_symbol:
        return dashboards

    try:
        ticker = yf.Ticker(ticker_symbol)
    except Exception as exc:
        logger.error(f"Failed to initialize yfinance ticker for dashboards ({ticker_symbol}): {exc}", exc_info=True)
        return dashboards

    intraday_df = pd.DataFrame()
    fast_info = {}
    try:
        intraday_raw = ticker.history(period="1d", interval="5m")
        intraday_df = _prepare_history_frame(intraday_raw)
    except Exception as exc:
        logger.warning(f"Unable to fetch intraday data for {ticker_symbol}: {exc}")

    try:
        fast_info = getattr(ticker, "fast_info", {}) or {}
    except Exception:
        fast_info = {}

    if not intraday_df.empty:
        latest_close = intraday_df["Close"].iloc[-1]
        previous_close = fast_info.get("previous_close")
        if previous_close in (None, 0) or (isinstance(previous_close, float) and math.isnan(previous_close)):
            if len(intraday_df) > 1:
                previous_close = intraday_df["Close"].iloc[0]
            else:
                previous_close = None
        change_value, change_percent = _compute_change(float(latest_close), float(previous_close) if previous_close not in (None, "") else None)
        live_series = _serialize_price_series(intraday_df.tail(60))
        dashboards["live"] = {
            "latest_price": _safe_round(latest_close),
            "previous_close": _safe_round(previous_close),
            "price_change": _safe_round(change_value),
            "price_change_percent": _safe_round(change_percent),
            "high": _safe_round(intraday_df["High"].max()),
            "low": _safe_round(intraday_df["Low"].min()),
            "volume": _safe_int(fast_info.get("last_volume")) or _safe_int(intraday_df["Volume"].iloc[-1]),
            "series": live_series,
            "interval": "5m"
        }
        day_change_value, day_change_percent = _compute_change(float(latest_close), float(intraday_df["Close"].iloc[0]))
        dashboards["one_day"] = {
            "latest_price": _safe_round(latest_close),
            "price_change": _safe_round(day_change_value),
            "price_change_percent": _safe_round(day_change_percent),
            "high": _safe_round(intraday_df["High"].max()),
            "low": _safe_round(intraday_df["Low"].min()),
            "volume": _safe_int(intraday_df["Volume"].sum()),
            "series": _serialize_price_series(intraday_df),
            "interval": "5m"
        }

    period_configs = {
        "five_day": {"period": "5d", "interval": "30m"},
        "one_month": {"period": "1mo", "interval": "1d"},
        "three_month": {"period": "3mo", "interval": "1d"}
    }

    for label, config in period_configs.items():
        try:
            raw_df = ticker.history(period=config["period"], interval=config["interval"])
        except Exception as exc:
            logger.warning(f"Failed to fetch history for {ticker_symbol} ({label}): {exc}")
            continue

        prepared_df = _prepare_history_frame(raw_df)
        if prepared_df.empty:
            continue

        closing_series = prepared_df["Close"]
        latest_price = closing_series.iloc[-1]
        start_price = closing_series.iloc[0]
        change_value, change_percent = _compute_change(float(latest_price), float(start_price))
        dashboards[label] = {
            "latest_price": _safe_round(latest_price),
            "price_change": _safe_round(change_value),
            "price_change_percent": _safe_round(change_percent),
            "high": _safe_round(prepared_df["High"].max()),
            "low": _safe_round(prepared_df["Low"].min()),
            "volume": _safe_int(prepared_df["Volume"].sum()),
            "series": _serialize_price_series(prepared_df),
            "interval": config["interval"]
        }

    return dashboards

def search_companies_by_query(query, region='US', lang='en-US', max_results=6):
    search_url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query,
        "lang": lang,
        "region": region,
        "quotesCount": max_results,
        "newsCount": 0
    }
    response = requests.get(
        search_url,
        params=params,
        timeout=5,
        headers=_build_yahoo_headers()
    )
    response.raise_for_status()
    payload = response.json()

    results = []
    for quote in payload.get("quotes", []):
        symbol = quote.get("symbol")
        if not symbol:
            continue
        results.append({
            "symbol": symbol,
            "name": quote.get("shortname") or quote.get("longname") or quote.get("name") or symbol,
            "exchange": quote.get("exchangeDisplay") or quote.get("fullExchangeName"),
            "assetType": quote.get("typeDisp") or quote.get("quoteType"),
            "sector": quote.get("sector"),
            "industry": quote.get("industry"),
        })
        if len(results) >= max_results:
            break
    return results

@app.route('/company_search')
def company_search():
    if not _require_authentication():
        return jsonify({"results": [], "error": "Unauthorized"}), 401
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({"results": []})
    region = request.args.get('region', 'US')
    lang = request.args.get('lang', 'en-US')
    try:
        matches = search_companies_by_query(query, region=region, lang=lang)
        return jsonify({"results": matches})
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response else 500
        logger.error(f"Yahoo Finance search HTTP error ({status_code}) for query '{query}': {http_err}")
        return jsonify({"error": "Upstream search service error.", "results": []}), status_code
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Network error querying Yahoo Finance search for '{query}': {req_err}")
        return jsonify({"error": "Failed to reach search service.", "results": []}), 503
    except Exception as exc:
        logger.error(f"Unexpected error during company search for '{query}': {exc}", exc_info=True)
        return jsonify({"error": "Internal error while searching companies.", "results": []}), 500


@app.route('/api/top-tech-stocks', methods=['POST'])
def api_top_tech_stocks():
    if not _require_authentication():
        return jsonify({"error": "Unauthorized"}), 401
    if not session.get(TOP_TECH_PASSWORD_SESSION_KEY):
        return jsonify({"error": "Access denied. Unlock the Top Tech Stocks page first."}), 403

    payload = request.get_json(silent=True) or {}
    requested_limit = payload.get("limit", 10)
    try:
        limit = int(requested_limit)
    except (ValueError, TypeError):
        limit = 10
    limit = max(1, min(limit, 25))

    try:
        rankings_payload = _compute_top_tech_rankings(limit=limit)
    except Exception as exc:
        logger.error(f"Failed to compute top tech rankings: {exc}", exc_info=True)
        return jsonify({"error": "Unable to compute rankings at this time."}), 500

    return jsonify(rankings_payload)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
