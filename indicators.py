"""
기술적 지표 계산 모듈
======================
data_pipeline.py 로 수집한 OHLCV DataFrame을 받아
HMM 입력 피처 + 트레이딩 신호용 지표를 계산합니다.

사용법:
    from indicators import add_indicators
    df = add_indicators(df)
"""

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame에 모든 기술적 지표를 추가하여 반환.
    HMM 피처 + RSI/MACD/ADX 포함.
    """
    df = df.copy()
    df = _add_moving_averages(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger(df)
    df = _add_adx(df)
    df = _add_atr(df)
    df = _add_volume_indicators(df)
    df = _add_hmm_features(df)
    return df.dropna()


# ── 이동평균 ─────────────────────────────────────────────────
def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for period in [5, 10, 20, 60, 120]:
        df[f"ma{period}"] = df["close"].rolling(period).mean()
        df[f"ma{period}_slope"] = df[f"ma{period}"].pct_change(5)  # 기울기

    # 골든/데드 크로스
    df["golden_cross"] = (df["ma20"] > df["ma60"]).astype(int)
    df["ma_trend"] = (df["ma20"] / df["ma60"] - 1) * 100  # % 괴리율
    return df


# ── RSI ──────────────────────────────────────────────────────
def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_signal"] = np.where(df["rsi"] > 70, -1, np.where(df["rsi"] < 30, 1, 0))
    return df


# ── MACD ─────────────────────────────────────────────────────
def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_cross"] = np.sign(df["macd_hist"]).diff().fillna(0)  # +2 골든, -2 데드
    return df


# ── 볼린저 밴드 ──────────────────────────────────────────────
def _add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    df["bb_upper"] = mid + std * sigma
    df["bb_lower"] = mid - std * sigma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid  # 변동성 지표
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


# ── ADX (추세 강도) ──────────────────────────────────────────
def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    dm_plus[dm_plus < dm_minus] = 0
    dm_minus[dm_minus < dm_plus] = 0

    atr = tr.ewm(span=period).mean()
    di_plus = 100 * dm_plus.ewm(span=period).mean() / atr
    di_minus = 100 * dm_minus.ewm(span=period).mean() / atr

    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    df["adx"] = dx.ewm(span=period).mean()
    df["di_plus"] = di_plus
    df["di_minus"] = di_minus
    df["adx_trend"] = (df["adx"] > 25).astype(int)  # 25 이상 = 추세 존재
    return df


# ── ATR (변동성) ─────────────────────────────────────────────
def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period).mean()
    df["atr_pct"] = df["atr"] / df["close"] * 100  # 정규화된 ATR
    return df


# ── 거래량 지표 ──────────────────────────────────────────────
def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]  # 1.5 이상 = 거래량 급증

    # OBV (On-Balance Volume)
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ma"] = obv.rolling(20).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma"]).astype(int)
    return df


# ── HMM 입력 피처 (3종) ──────────────────────────────────────
def _add_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HMM 모델에 직접 들어가는 핵심 피처.
    방식 1에서 언급한 returns / range / vol_change 기반.
    """
    # 정규화 (Z-score, 롤링 252일)
    for col in ["returns", "range", "vol_change"]:
        rolling_mean = df[col].rolling(252).mean()
        rolling_std = df[col].rolling(252).std()
        df[f"{col}_z"] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)

    # 복합 모멘텀 스코어 (1M / 3M / 6M)
    df["mom_1m"] = df["close"].pct_change(21)
    df["mom_3m"] = df["close"].pct_change(63)
    df["mom_6m"] = df["close"].pct_change(126)
    df["mom_score"] = (df["mom_1m"] + df["mom_3m"] + df["mom_6m"]) / 3

    return df


# ── 매매 신호 집계 ───────────────────────────────────────────
def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    최신 행 기준으로 각 지표의 매수/매도 신호를 집계.
    HMM과 함께 '몇 개 일치' 조건 판단에 활용.
    """
    if df.empty:
        return {}

    row = df.iloc[-1]
    signals = {}

    # RSI
    signals["rsi"] = "buy" if row.get("rsi", 50) < 30 else ("sell" if row.get("rsi", 50) > 70 else "neutral")

    # MACD
    signals["macd"] = "buy" if row.get("macd_hist", 0) > 0 else "sell"

    # ADX 추세
    signals["adx_trend"] = "strong" if row.get("adx", 0) > 25 else "weak"
    signals["adx_direction"] = "buy" if row.get("di_plus", 0) > row.get("di_minus", 0) else "sell"

    # 볼린저
    bb_pct = row.get("bb_pct", 0.5)
    signals["bollinger"] = "buy" if bb_pct < 0.2 else ("sell" if bb_pct > 0.8 else "neutral")

    # MA 골든크로스
    signals["ma_cross"] = "buy" if row.get("golden_cross", 0) == 1 else "sell"

    # 거래량
    signals["volume"] = "surge" if row.get("vol_ratio", 1) > 1.5 else "normal"

    # 모멘텀
    mom = row.get("mom_score", 0)
    signals["momentum"] = "buy" if mom > 0.03 else ("sell" if mom < -0.03 else "neutral")

    # 집계
    buy_count = sum(1 for v in signals.values() if v == "buy")
    sell_count = sum(1 for v in signals.values() if v == "sell")

    return {
        "signals": signals,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "overall": "buy" if buy_count >= 4 else ("sell" if sell_count >= 4 else "neutral"),
        "rsi": round(row.get("rsi", 0), 2),
        "adx": round(row.get("adx", 0), 2),
        "macd_hist": round(row.get("macd_hist", 0), 4),
        "bb_pct": round(row.get("bb_pct", 0), 3),
        "mom_score": round(row.get("mom_score", 0), 4),
    }


if __name__ == "__main__":
    # 테스트용 랜덤 데이터
    np.random.seed(42)
    n = 400
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    dates = pd.date_range("2023-01-01", periods=n)
    demo = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.003),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close":  close,
        "volume": np.abs(np.random.randn(n)) * 1e6 + 5e5,
        "returns": np.random.randn(n) * 0.01,
        "log_returns": np.random.randn(n) * 0.01,
        "range": np.abs(np.random.randn(n)) * 0.01,
        "vol_change": np.random.randn(n) * 0.1,
    }, index=dates)

    result = add_indicators(demo)
    summary = get_signal_summary(result)

    print(f"지표 추가 후 컬럼 수: {len(result.columns)}")
    print(f"\n현재 신호 요약:")
    for k, v in summary.items():
        if k != "signals":
            print(f"  {k}: {v}")
    print(f"\n개별 신호: {summary['signals']}")
