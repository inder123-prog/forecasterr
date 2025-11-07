from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternMeta:
    name: str
    type: str  # bullish, bearish, continuation
    weight: float


PATTERN_METADATA: Dict[str, PatternMeta] = {
    "pattern_hammer": PatternMeta("Hammer", "bullish", 1.5),
    "pattern_inverted_hammer": PatternMeta("Inverted Hammer", "bullish", 1.2),
    "pattern_bullish_engulfing": PatternMeta("Bullish Engulfing", "bullish", 2.0),
    "pattern_piercing_line": PatternMeta("Piercing Line", "bullish", 1.7),
    "pattern_morning_star": PatternMeta("Morning Star", "bullish", 2.3),
    "pattern_three_white_soldiers": PatternMeta("Three White Soldiers", "bullish", 2.8),
    "pattern_hanging_man": PatternMeta("Hanging Man", "bearish", 1.5),
    "pattern_shooting_star": PatternMeta("Shooting Star", "bearish", 1.2),
    "pattern_bearish_engulfing": PatternMeta("Bearish Engulfing", "bearish", 2.0),
    "pattern_evening_star": PatternMeta("Evening Star", "bearish", 2.3),
    "pattern_three_black_crows": PatternMeta("Three Black Crows", "bearish", 2.8),
    "pattern_dark_cloud_cover": PatternMeta("Dark Cloud Cover", "bearish", 1.8),
    "pattern_doji": PatternMeta("Doji", "continuation", 1.0),
    "pattern_spinning_top": PatternMeta("Spinning Top", "continuation", 1.0),
    "pattern_falling_three_methods": PatternMeta("Falling Three Methods", "continuation", 1.6),
    "pattern_rising_three_methods": PatternMeta("Rising Three Methods", "continuation", 1.6),
}


def analyze_candlestick_patterns(ohlc_df: pd.DataFrame) -> Dict[str, object]:
    """
    Detect candlestick patterns on the provided OHLC dataframe.

    Returns a dictionary with two items:
        - pattern_features: DataFrame with a boolean/int column per pattern along with aggregate scores.
        - summary: dictionary describing the latest detected signals and aggregate sentiment.
    """
    empty_payload = {
        "pattern_features": pd.DataFrame(columns=["ds"] + list(PATTERN_METADATA.keys()) + [
            "pattern_bullish_score", "pattern_bearish_score", "pattern_continuation_score", "pattern_net_score"
        ]),
        "summary": {
            "latest_signals": [],
            "recent_signals": [],
            "bullish_score": 0.0,
            "bearish_score": 0.0,
            "continuation_score": 0.0,
            "net_score": 0.0,
            "bias": "neutral",
        },
    }

    if ohlc_df is None or ohlc_df.empty:
        return empty_payload

    df = _prepare_dataframe(ohlc_df)
    if df is None or df.empty:
        return empty_payload

    base = _compute_base_columns(df)
    patterns = {}

    patterns["pattern_hammer"] = _detect_hammer(base)
    patterns["pattern_inverted_hammer"] = _detect_inverted_hammer(base)
    patterns["pattern_bullish_engulfing"] = _detect_bullish_engulfing(base)
    patterns["pattern_piercing_line"] = _detect_piercing_line(base)
    patterns["pattern_morning_star"] = _detect_morning_star(base)
    patterns["pattern_three_white_soldiers"] = _detect_three_white_soldiers(base)

    patterns["pattern_hanging_man"] = _detect_hanging_man(base)
    patterns["pattern_shooting_star"] = _detect_shooting_star(base)
    patterns["pattern_bearish_engulfing"] = _detect_bearish_engulfing(base)
    patterns["pattern_evening_star"] = _detect_evening_star(base)
    patterns["pattern_three_black_crows"] = _detect_three_black_crows(base)
    patterns["pattern_dark_cloud_cover"] = _detect_dark_cloud_cover(base)

    patterns["pattern_doji"] = _detect_doji(base)
    patterns["pattern_spinning_top"] = _detect_spinning_top(base)
    patterns["pattern_falling_three_methods"] = _detect_falling_three_methods(base)
    patterns["pattern_rising_three_methods"] = _detect_rising_three_methods(base)

    pattern_df = pd.DataFrame({"ds": base["ds"]})
    for key, series in patterns.items():
        pattern_df[key] = series.astype(int)

    bullish_score = np.zeros(len(pattern_df))
    bearish_score = np.zeros(len(pattern_df))
    continuation_score = np.zeros(len(pattern_df))

    for key, meta in PATTERN_METADATA.items():
        weights = pattern_df[key].to_numpy() * meta.weight
        if meta.type == "bullish":
            bullish_score += weights
        elif meta.type == "bearish":
            bearish_score += weights
        else:
            continuation_score += weights

    pattern_df["pattern_bullish_score"] = bullish_score
    pattern_df["pattern_bearish_score"] = bearish_score
    pattern_df["pattern_continuation_score"] = continuation_score
    pattern_df["pattern_net_score"] = bullish_score - bearish_score

    summary = _build_summary(pattern_df)

    return {
        "pattern_features": pattern_df,
        "summary": summary,
    }


def _prepare_dataframe(ohlc_df: pd.DataFrame) -> pd.DataFrame | None:
    df = ohlc_df.copy()
    rename_map = {col: col.lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    if "ds" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ds"})
        else:
            return None

    required_cols = {"ds", "open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return None

    df = df[list(required_cols)].copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=False).dt.tz_localize(None)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def _compute_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["body"] = (base["close"] - base["open"]).abs()
    base["upper_shadow"] = base["high"] - np.maximum(base["open"], base["close"])
    base["lower_shadow"] = np.minimum(base["open"], base["close"]) - base["low"]
    base["range"] = base["high"] - base["low"]
    range_safe = base["range"].replace(0, np.nan)

    base["body_ratio"] = (base["body"] / range_safe).fillna(0.0)
    base["upper_ratio"] = (base["upper_shadow"] / range_safe).fillna(0.0)
    base["lower_ratio"] = (base["lower_shadow"] / range_safe).fillna(0.0)

    base["is_bullish"] = base["close"] > base["open"]
    base["is_bearish"] = base["close"] < base["open"]
    base["avg_body"] = base["body"].rolling(window=10, min_periods=2).mean()

    base["downtrend"] = (
        (base["close"].shift(1) < base["close"].shift(2))
        & (base["close"].shift(2) < base["close"].shift(3))
    ).fillna(False)
    base["strong_downtrend"] = (
        base["downtrend"] & (base["close"].shift(3) < base["close"].shift(4))
    ).fillna(False)

    base["uptrend"] = (
        (base["close"].shift(1) > base["close"].shift(2))
        & (base["close"].shift(2) > base["close"].shift(3))
    ).fillna(False)
    base["strong_uptrend"] = (
        base["uptrend"] & (base["close"].shift(3) > base["close"].shift(4))
    ).fillna(False)

    return base


def _detect_hammer(base: pd.DataFrame) -> pd.Series:
    return (
        base["strong_downtrend"]
        & (base["lower_shadow"] >= 2 * base["body"])
        & (base["upper_shadow"] <= base["body"])
        & (base["body_ratio"] <= 0.4)
    )


def _detect_inverted_hammer(base: pd.DataFrame) -> pd.Series:
    return (
        base["strong_downtrend"]
        & (base["upper_shadow"] >= 2 * base["body"])
        & (base["lower_shadow"] <= base["body"] * 0.3)
        & (base["body_ratio"] <= 0.4)
    )


def _detect_bullish_engulfing(base: pd.DataFrame) -> pd.Series:
    prev = base.shift(1)
    return (
        base["downtrend"]
        & prev["is_bearish"]
        & base["is_bullish"]
        & (base["body"] > prev["body"])
        & (base["close"] >= prev[["open", "close"]].max(axis=1))
        & (base["open"] <= prev[["open", "close"]].min(axis=1))
    ).fillna(False)


def _detect_piercing_line(base: pd.DataFrame) -> pd.Series:
    prev = base.shift(1)
    midpoint = (prev["open"] + prev["close"]) / 2
    return (
        base["downtrend"]
        & prev["is_bearish"]
        & base["is_bullish"]
        & (prev["body_ratio"] >= 0.6)
        & (base["open"] < prev["low"])
        & (base["close"] > midpoint)
        & (base["close"] < prev["open"])
    ).fillna(False)


def _detect_morning_star(base: pd.DataFrame) -> pd.Series:
    curr = base
    prev1 = base.shift(1)
    prev2 = base.shift(2)
    return (
        curr["downtrend"].shift(1).fillna(False)
        & prev2["is_bearish"]
        & (prev2["body_ratio"] >= 0.6)
        & (prev1["body_ratio"] <= 0.35)
        & (prev1["low"] < prev2["low"])
        & curr["is_bullish"]
        & (curr["body_ratio"] >= 0.5)
        & (curr["close"] > (prev2["open"] + prev2["close"]) / 2)
    ).fillna(False)


def _detect_three_white_soldiers(base: pd.DataFrame) -> pd.Series:
    one = base.shift(2)
    two = base.shift(1)
    three = base
    return (
        three["downtrend"].shift(1).fillna(False)
        & one["is_bullish"]
        & two["is_bullish"]
        & three["is_bullish"]
        & (one["body_ratio"] >= 0.5)
        & (two["body_ratio"] >= 0.5)
        & (three["body_ratio"] >= 0.5)
        & (two["open"] > one["open"])
        & (two["open"] < one["close"])
        & (three["open"] > two["open"])
        & (three["open"] < two["close"])
        & (two["close"] > one["close"])
        & (three["close"] > two["close"])
    ).fillna(False)


def _detect_hanging_man(base: pd.DataFrame) -> pd.Series:
    return (
        base["strong_uptrend"]
        & (base["lower_shadow"] >= 2 * base["body"])
        & (base["upper_shadow"] <= base["body"])
        & (base["body_ratio"] <= 0.4)
    )


def _detect_shooting_star(base: pd.DataFrame) -> pd.Series:
    return (
        base["strong_uptrend"]
        & (base["upper_shadow"] >= 2 * base["body"])
        & (base["lower_shadow"] <= base["body"] * 0.3)
        & (base["body_ratio"] <= 0.4)
    )


def _detect_bearish_engulfing(base: pd.DataFrame) -> pd.Series:
    prev = base.shift(1)
    return (
        base["uptrend"]
        & prev["is_bullish"]
        & base["is_bearish"]
        & (base["body"] > prev["body"])
        & (base["close"] <= prev[["open", "close"]].min(axis=1))
        & (base["open"] >= prev[["open", "close"]].max(axis=1))
    ).fillna(False)


def _detect_evening_star(base: pd.DataFrame) -> pd.Series:
    curr = base
    prev1 = base.shift(1)
    prev2 = base.shift(2)
    return (
        curr["uptrend"].shift(1).fillna(False)
        & prev2["is_bullish"]
        & (prev2["body_ratio"] >= 0.6)
        & (prev1["body_ratio"] <= 0.35)
        & (prev1["high"] > prev2["high"])
        & curr["is_bearish"]
        & (curr["body_ratio"] >= 0.5)
        & (curr["close"] < (prev2["open"] + prev2["close"]) / 2)
    ).fillna(False)


def _detect_three_black_crows(base: pd.DataFrame) -> pd.Series:
    one = base.shift(2)
    two = base.shift(1)
    three = base
    return (
        three["uptrend"].shift(1).fillna(False)
        & one["is_bearish"]
        & two["is_bearish"]
        & three["is_bearish"]
        & (one["body_ratio"] >= 0.5)
        & (two["body_ratio"] >= 0.5)
        & (three["body_ratio"] >= 0.5)
        & (two["open"] < one["open"])
        & (two["open"] > one["close"])
        & (three["open"] < two["open"])
        & (three["open"] > two["close"])
        & (two["close"] < one["close"])
        & (three["close"] < two["close"])
    ).fillna(False)


def _detect_dark_cloud_cover(base: pd.DataFrame) -> pd.Series:
    prev = base.shift(1)
    midpoint = (prev["open"] + prev["close"]) / 2
    return (
        base["uptrend"]
        & prev["is_bullish"]
        & base["is_bearish"]
        & (prev["body_ratio"] >= 0.6)
        & (base["open"] > prev["high"])
        & (base["close"] < midpoint)
        & (base["close"] > prev["open"])
    ).fillna(False)


def _detect_doji(base: pd.DataFrame) -> pd.Series:
    return ((base["body_ratio"] <= 0.1) & (base["range"] > 0)).fillna(False)


def _detect_spinning_top(base: pd.DataFrame) -> pd.Series:
    return (
        (base["body_ratio"] <= 0.4)
        & (base["upper_ratio"] >= 0.2)
        & (base["lower_ratio"] >= 0.2)
        & (base["range"] > 0)
    ).fillna(False)


def _detect_falling_three_methods(base: pd.DataFrame) -> pd.Series:
    c1 = base.shift(4)
    c2 = base.shift(3)
    c3 = base.shift(2)
    c4 = base.shift(1)
    c5 = base
    return (
        base["downtrend"].shift(2).fillna(False)
        & c1["is_bearish"]
        & (c1["body_ratio"] >= 0.6)
        & c2["is_bullish"]
        & c3["is_bullish"]
        & c4["is_bullish"]
        & (c2["body_ratio"] <= 0.4)
        & (c3["body_ratio"] <= 0.4)
        & (c4["body_ratio"] <= 0.4)
        & (c2["close"] <= c1["open"])
        & (c2["close"] >= c1["close"])
        & (c3["close"] <= c1["open"])
        & (c3["close"] >= c1["close"])
        & (c4["close"] <= c1["open"])
        & (c4["close"] >= c1["close"])
        & c5["is_bearish"]
        & (c5["close"] < c1["close"])
    ).fillna(False)


def _detect_rising_three_methods(base: pd.DataFrame) -> pd.Series:
    c1 = base.shift(4)
    c2 = base.shift(3)
    c3 = base.shift(2)
    c4 = base.shift(1)
    c5 = base
    return (
        base["uptrend"].shift(2).fillna(False)
        & c1["is_bullish"]
        & (c1["body_ratio"] >= 0.6)
        & c2["is_bearish"]
        & c3["is_bearish"]
        & c4["is_bearish"]
        & (c2["body_ratio"] <= 0.4)
        & (c3["body_ratio"] <= 0.4)
        & (c4["body_ratio"] <= 0.4)
        & (c2["close"] >= c1["open"])
        & (c2["close"] <= c1["close"])
        & (c3["close"] >= c1["open"])
        & (c3["close"] <= c1["close"])
        & (c4["close"] >= c1["open"])
        & (c4["close"] <= c1["close"])
        & c5["is_bullish"]
        & (c5["close"] > c1["close"])
    ).fillna(False)


def _build_summary(pattern_df: pd.DataFrame) -> Dict[str, object]:
    if pattern_df.empty:
        return {
            "latest_signals": [],
            "recent_signals": [],
            "bullish_score": 0.0,
            "bearish_score": 0.0,
            "continuation_score": 0.0,
            "net_score": 0.0,
            "bias": "neutral",
        }

    recent_window = pattern_df.tail(15)
    pattern_columns = list(PATTERN_METADATA.keys())

    recent_signals: List[Dict[str, object]] = []
    for _, row in recent_window.iterrows():
        ds_val = row["ds"]
        for col in pattern_columns:
            if row[col]:
                meta = PATTERN_METADATA[col]
                recent_signals.append(
                    {
                        "date": ds_val.strftime("%Y-%m-%d"),
                        "pattern": meta.name,
                        "type": meta.type,
                        "weight": meta.weight,
                    }
                )

    latest_row = pattern_df.tail(1).iloc[0]
    latest_signals = [
        signal for signal in recent_signals if signal["date"] == latest_row["ds"].strftime("%Y-%m-%d")
    ]

    bullish_score = float(pattern_df["pattern_bullish_score"].tail(5).sum())
    bearish_score = float(pattern_df["pattern_bearish_score"].tail(5).sum())
    continuation_score = float(pattern_df["pattern_continuation_score"].tail(5).sum())
    net_score = bullish_score - bearish_score

    if net_score > 0.5:
        bias = "bullish"
    elif net_score < -0.5:
        bias = "bearish"
    elif continuation_score > max(bullish_score, bearish_score):
        bias = "sideways"
    else:
        bias = "neutral"

    return {
        "latest_signals": latest_signals,
        "recent_signals": recent_signals,
        "bullish_score": round(bullish_score, 2),
        "bearish_score": round(bearish_score, 2),
        "continuation_score": round(continuation_score, 2),
        "net_score": round(net_score, 2),
        "bias": bias,
    }

