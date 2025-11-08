from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - best effort fallback
    LogisticRegression = None  # type: ignore
    StandardScaler = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingExplanation:
    feature: str
    direction: str
    weight: float
    value: float


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def _prepare_feature_table(ohlc_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if ohlc_df is None or ohlc_df.empty:
        return pd.DataFrame(), None

    df = ohlc_df.copy()
    rename_map = {col: col.lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    if "ds" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ds"})
        else:
            return pd.DataFrame(), None

    required_cols = {"ds", "open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame(), None

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.sort_values("ds").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df.get("volume", pd.Series(dtype=float)).fillna(0.0)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    feature_table = pd.DataFrame({"ds": df["ds"], "close": close})
    feature_table["return_1"] = close.pct_change()
    feature_table["return_5"] = close.pct_change(periods=5)
    feature_table["return_10"] = close.pct_change(periods=10)
    feature_table["volatility_14"] = feature_table["return_1"].rolling(window=14).std()
    feature_table["volatility_30"] = feature_table["return_1"].rolling(window=30).std()

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    feature_table["ema_fast"] = ema_fast
    feature_table["ema_slow"] = ema_slow
    feature_table["macd"] = macd
    feature_table["macd_signal"] = macd_signal
    feature_table["macd_hist"] = macd - macd_signal
    feature_table["ema_distance"] = (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)

    rsi = _compute_rsi(close, period=14)
    feature_table["rsi_14"] = rsi / 100.0  # scale 0-1

    atr = _compute_atr(high, low, close, period=14)
    feature_table["atr_14"] = atr
    feature_table["atr_ratio"] = atr / close.replace(0.0, np.nan)

    volume = df["volume"].replace(0.0, np.nan)
    feature_table["volume_zscore"] = (
        (volume - volume.rolling(window=20, min_periods=5).mean())
        / volume.rolling(window=20, min_periods=5).std()
    ).fillna(0.0)

    feature_table["momentum_rank"] = feature_table["return_5"].rank(pct=True)
    feature_table["forward_return_1"] = close.shift(-1) / close - 1
    feature_table["target"] = (feature_table["forward_return_1"] > 0).astype(float)

    latest_row = feature_table.tail(1).iloc[0]
    latest_features = latest_row.drop(labels=["forward_return_1", "target"], errors="ignore")

    train_table = feature_table.iloc[:-1].dropna()
    if train_table.empty:
        return pd.DataFrame(), latest_features

    return train_table, latest_features


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _build_explanations(
    feature_names: List[str],
    coefficients: np.ndarray,
    latest_vector: np.ndarray,
    top_k: int = 5,
) -> List[TradingExplanation]:
    if len(feature_names) != len(coefficients):
        return []

    abs_weights = np.abs(coefficients)
    if np.all(abs_weights == 0):
        return []

    indices = abs_weights.argsort()[::-1][:top_k]
    explanations: List[TradingExplanation] = []
    weight_distribution = _softmax(abs_weights[indices])

    for rank, idx in enumerate(indices):
        coeff = float(coefficients[idx])
        value = float(latest_vector[idx])
        direction = "bullish" if coeff * value >= 0 else "bearish"
        explanations.append(
            TradingExplanation(
                feature=feature_names[idx],
                direction=direction,
                weight=float(weight_distribution[rank]),
                value=value,
            )
        )
    return explanations


def _summarize_backtest(
    ds_series: pd.Series,
    close_series: pd.Series,
    probs: np.ndarray,
    forward_returns: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
) -> Dict[str, object]:
    signals = np.where(probs >= buy_threshold, 1, np.where(probs <= sell_threshold, -1, 0))
    strategy_returns = forward_returns.to_numpy() * signals

    trade_mask = signals != 0
    active_returns = strategy_returns[trade_mask]
    trade_dates = ds_series.to_numpy()[trade_mask]
    trade_probs = probs[trade_mask]
    trade_signals = signals[trade_mask]
    trade_forward_returns = forward_returns.to_numpy()[trade_mask]

    cumulative_curve = np.cumprod(1 + strategy_returns) if strategy_returns.size else np.array([])
    if cumulative_curve.size:
        peaks = np.maximum.accumulate(cumulative_curve)
        drawdowns = (cumulative_curve - peaks) / peaks
        max_drawdown = float(drawdowns.min())
        strategy_return = float(cumulative_curve[-1] - 1)
    else:
        max_drawdown = 0.0
        strategy_return = 0.0

    if close_series.empty:
        buy_hold_return = 0.0
    else:
        buy_hold_return = float(close_series.iloc[-1] / close_series.iloc[0] - 1)

    hit_count = 0
    for signal, realized in zip(trade_signals, trade_forward_returns):
        if signal == 1 and realized > 0:
            hit_count += 1
        elif signal == -1 and realized < 0:
            hit_count += 1
    total_trades = int(trade_mask.sum())
    hit_rate = float(hit_count / total_trades) if total_trades else 0.0
    avg_trade_return = float(active_returns.mean()) if active_returns.size else 0.0

    last_trades: List[Dict[str, object]] = []
    for date, signal, prob, realized in list(zip(trade_dates, trade_signals, trade_probs, trade_forward_returns))[-5:]:
        last_trades.append(
            {
                "date": date.isoformat() if isinstance(date, pd.Timestamp) else str(date),
                "signal": "buy" if signal == 1 else "sell",
                "probability": float(prob),
                "next_day_return": float(realized),
            }
        )

    return {
        "evaluation_window": int(len(ds_series)),
        "trades": total_trades,
        "hit_rate": hit_rate,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_drawdown,
        "avg_trade_return": avg_trade_return,
        "last_signals": last_trades,
    }


def generate_ai_trading_plan(
    ticker: str,
    ohlc_df: pd.DataFrame,
    current_price: Optional[float],
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> Dict[str, object]:
    """
    Build an AI-assisted trading plan using a logistic-regression classifier on engineered signals.
    Returns a JSON-serialisable payload that includes the latest recommendation,
    feature explanations, and a simple backtest summary.
    """
    if LogisticRegression is None or StandardScaler is None:
        logger.warning("scikit-learn is unavailable; skipping AI trading plan generation.")
        return {
            "status": "unavailable",
            "reason": "scikit-learn not installed",
        }

    feature_table, latest_features = _prepare_feature_table(ohlc_df)
    if feature_table.empty or latest_features is None:
        logger.info("Insufficient OHLC history to train AI trading model for %s.", ticker)
        return {
            "status": "insufficient_data",
            "reason": "Not enough enriched OHLC data to build signals.",
        }

    if len(feature_table) < 120:
        logger.info("Not enough rows (%s) to build AI trading plan for %s.", len(feature_table), ticker)
        return {
            "status": "insufficient_data",
            "reason": "At least 120 enriched rows are required.",
        }

    window = 40 if len(feature_table) >= 200 else 20
    train_df = feature_table.iloc[:-window]
    test_df = feature_table.iloc[-window:]

    if train_df.empty or test_df.empty:
        return {
            "status": "insufficient_data",
            "reason": "Unable to split training and evaluation windows.",
        }

    feature_columns = [
        col
        for col in train_df.columns
        if col
        not in {
            "ds",
            "target",
            "forward_return_1",
            "close",
        }
    ]
    if not feature_columns:
        return {
            "status": "insufficient_data",
            "reason": "No usable feature columns detected.",
        }

    try:
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")

        X_train = scaler.fit_transform(train_df[feature_columns])
        y_train = train_df["target"].astype(int)
        model.fit(X_train, y_train)

        X_test = scaler.transform(test_df[feature_columns])
        y_test = test_df["target"].astype(int)
        test_probs = model.predict_proba(X_test)[:, 1]

        train_accuracy = float(model.score(X_train, y_train))
        test_accuracy = float(model.score(X_test, y_test))
    except Exception as exc:  # pragma: no cover - robustness
        logger.error("Failed to train AI trading model for %s: %s", ticker, exc, exc_info=True)
        return {
            "status": "error",
            "reason": f"Training error: {exc}",
        }

    latest_vector = latest_features[feature_columns].to_numpy(dtype=float)
    try:
        latest_scaled = scaler.transform(latest_vector.reshape(1, -1))
    except Exception as exc:
        logger.error("Failed to scale latest feature vector for %s: %s", ticker, exc, exc_info=True)
        return {
            "status": "error",
            "reason": f"Scaling error: {exc}",
        }

    latest_prob_up = float(model.predict_proba(latest_scaled)[0, 1])
    latest_prob_down = 1.0 - latest_prob_up
    confidence = float(abs(latest_prob_up - 0.5) * 2)

    if latest_prob_up >= buy_threshold:
        action = "buy"
    elif latest_prob_up <= sell_threshold:
        action = "sell"
    else:
        action = "hold"

    volatility = float(latest_features.get("volatility_14", np.nan))
    atr_value = float(latest_features.get("atr_14", np.nan))
    atr_ratio = float(latest_features.get("atr_ratio", np.nan))
    expected_return = float(latest_features.get("return_5", np.nan) / 5.0)

    stop_loss = None
    take_profit = None
    position_size = None
    if current_price and isinstance(current_price, (int, float)):
        if np.isfinite(atr_value) and atr_value > 0:
            stop_loss = float(current_price - 1.2 * atr_value)
            take_profit = float(current_price + 2.0 * atr_value)
            risk_fraction = atr_value / current_price if current_price else 0.02
            if risk_fraction > 0:
                position_size = float(min(0.25, max(0.05, 0.02 / risk_fraction)))
        elif np.isfinite(volatility) and volatility > 0:
            stop_loss = float(current_price * (1 - 2.5 * volatility))
            take_profit = float(current_price * (1 + 4.0 * volatility))

    explanations = _build_explanations(
        feature_columns,
        model.coef_[0],
        latest_vector,
    )

    backtest_summary = _summarize_backtest(
        ds_series=test_df["ds"],
        close_series=test_df["close"],
        probs=test_probs,
        forward_returns=test_df["forward_return_1"],
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    return {
        "status": "ready",
        "latest_signal": {
            "action": action,
            "probability_up": latest_prob_up,
            "probability_down": latest_prob_down,
            "confidence": confidence,
            "expected_daily_return": expected_return,
            "volatility_14": volatility if np.isfinite(volatility) else None,
            "atr_14": atr_value if np.isfinite(atr_value) else None,
            "atr_ratio": atr_ratio if np.isfinite(atr_ratio) else None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size_fraction": position_size,
            "reasoning": [
                explanation.__dict__
                for explanation in explanations
            ],
        },
        "model_summary": {
            "training_rows": int(len(train_df)),
            "evaluation_rows": int(len(test_df)),
            "train_accuracy": train_accuracy,
            "evaluation_accuracy": test_accuracy,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "feature_columns": feature_columns,
        },
        "backtest": backtest_summary,
    }
