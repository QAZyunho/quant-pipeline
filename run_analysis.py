"""
일별 자동 분석 스크립트
========================
GitHub Actions에서 매일 실행되어 분석 리포트를 생성합니다.
결과물: reports/YYYY-MM-DD.md
"""

import json
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── 로깅 ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 경로 ─────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
MD_DIR   = REPORTS_DIR / "md"
JSON_DIR = REPORTS_DIR / "json"
MD_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.today().strftime("%Y-%m-%d")
REPORT_PATH = MD_DIR / f"{TODAY}.md"
# ── 관심 종목 (watchlist.json에서 로드) ──────────────────────
WATCHLIST_PATH = Path("watchlist.json")

def load_watchlist() -> dict:
    if not WATCHLIST_PATH.exists():
        log.warning("watchlist.json 없음 → 기본 종목 사용")
        return {
            "KR": {"005930": "삼성전자"},
            "US": {"SPY": "S&P500 ETF"},
        }
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    kr_count = len(data.get("KR", {}))
    us_count = len(data.get("US", {}))
    log.info(f"watchlist 로드: 한국 {kr_count}개, 미국 {us_count}개")
    return data

WATCHLIST = load_watchlist()


# ════════════════════════════════════════════════════════════
# 데이터 수집
# ════════════════════════════════════════════════════════════
def fetch_kr(ticker: str, years: int = 2) -> pd.DataFrame:
    from pykrx import stock
    end   = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv(start, end, ticker)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df.columns = ["open", "high", "low", "close", "volume", "change_pct"]
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["range"]       = (df["high"] - df["low"]) / df["close"]
    df["vol_change"]  = df["volume"].pct_change()
    return df.dropna()

def fetch_us(ticker: str, years: int = 2) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["range"]       = (df["high"] - df["low"]) / df["close"]
    df["vol_change"]  = df["volume"].pct_change()
    return df.dropna()


# ════════════════════════════════════════════════════════════
# 지표 계산 (indicators.py 인라인 버전)
# ════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 이동평균
    for p in [5, 20, 60]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    df["golden_cross"] = (df["ma20"] > df["ma60"]).astype(int)
    df["ma_trend"]     = (df["ma20"] / df["ma60"] - 1) * 100

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # 볼린저
    mid = df["close"].rolling(20).mean()
    sig = df["close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * sig
    df["bb_lower"] = mid - 2 * sig
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

    # ADX
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    dm_p = high.diff().clip(lower=0)
    dm_m = (-low.diff()).clip(lower=0)
    dm_p[dm_p < dm_m] = 0
    dm_m[dm_m < dm_p] = 0
    atr   = tr.ewm(span=14).mean()
    di_p  = 100 * dm_p.ewm(span=14).mean() / atr
    di_m  = 100 * dm_m.ewm(span=14).mean() / atr
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["adx"]      = dx.ewm(span=14).mean()
    df["di_plus"]  = di_p
    df["di_minus"] = di_m

    # 모멘텀
    df["mom_1m"] = df["close"].pct_change(21)
    df["mom_3m"] = df["close"].pct_change(63)
    df["mom_6m"] = df["close"].pct_change(126)
    df["mom_score"] = (df["mom_1m"] + df["mom_3m"] + df["mom_6m"]) / 3

    # 거래량
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # HMM 피처 (정규화)
    for col in ["returns", "range", "vol_change"]:
        mu  = df[col].rolling(252).mean()
        std = df[col].rolling(252).std()
        df[f"{col}_z"] = (df[col] - mu) / std.replace(0, np.nan)

    return df.dropna()


def get_signal(df: pd.DataFrame) -> dict:
    """최신 행 기준 매수/매도 신호 집계."""
    r = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else r

    signals = {
        "RSI":        "매수" if r["rsi"] < 35 else ("매도" if r["rsi"] > 65 else "중립"),
        "MACD":       "매수" if r["macd_hist"] > 0 else "매도",
        "ADX방향":    "매수" if r["di_plus"] > r["di_minus"] else "매도",
        "볼린저":     "매수" if r["bb_pct"] < 0.25 else ("매도" if r["bb_pct"] > 0.75 else "중립"),
        "MA크로스":   "매수" if r["golden_cross"] == 1 else "매도",
        "모멘텀":     "매수" if r["mom_score"] > 0.02 else ("매도" if r["mom_score"] < -0.02 else "중립"),
        "거래량":     "급등" if r["vol_ratio"] > 1.5 else "보통",
    }
    buy  = sum(1 for v in signals.values() if v == "매수")
    sell = sum(1 for v in signals.values() if v == "매도")

    if buy >= 5:   overall = "🟢 강한매수"
    elif buy >= 3: overall = "🔵 약한매수"
    elif sell >= 5: overall = "🔴 강한매도"
    elif sell >= 3: overall = "🟠 약한매도"
    else:           overall = "⚪ 중립"

    return {
        "signals":   signals,
        "buy_count": buy,
        "sell_count": sell,
        "overall":   overall,
        "rsi":       round(float(r["rsi"]), 1),
        "adx":       round(float(r["adx"]), 1),
        "macd_hist": round(float(r["macd_hist"]), 4),
        "bb_pct":    round(float(r["bb_pct"]), 3),
        "mom_score": round(float(r["mom_score"]) * 100, 2),
        "vol_ratio": round(float(r["vol_ratio"]), 2),
        "close":     float(r["close"]),
        "returns_1d": round(float(r["returns"]) * 100, 2),
        # HMM 피처
        "returns_z":    round(float(r["returns_z"]), 2),
        "range_z":      round(float(r["range_z"]), 2),
        "vol_change_z": round(float(r["vol_change_z"]), 2),
    }


# ════════════════════════════════════════════════════════════
# 리포트 생성
# ════════════════════════════════════════════════════════════
def combined_row(ticker: str, name: str, sig: dict) -> str:
    """신호 + HMM 피처 통합 테이블 한 줄."""
    rz  = sig["returns_z"]
    raz = sig["range_z"]
    vcz = sig["vol_change_z"]

    if rz > 1.0 and raz < 0.5:
        regime = "🐂 강세"
    elif rz < -1.5:
        regime = "💥 폭락"
    elif raz > 1.5:
        regime = "⚡ 고변동"
    elif abs(rz) < 0.3 and raz < 0.3:
        regime = "😴 횡보"
    else:
        regime = "🌀 노이즈"

    return (
        f"| {ticker} | {name} | {sig['overall']} | "
        f"{sig['close']:,.0f} | {sig['returns_1d']:+.2f}% | "
        f"{sig['rsi']} | {sig['adx']} | {sig['mom_score']:+.2f}% | "
        f"{sig['buy_count']}/{sig['sell_count']} | "
        f"{rz:+.2f} | {raz:+.2f} | {vcz:+.2f} | {regime} |"
    )

def build_report(kr_results: dict, us_results: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    lines = [
        "# 📊 퀀트 일별 분석 리포트",
        "",
        f"> 생성일시: {now}  ",
        "> ⚠️ 본 리포트는 **보조 참고용**입니다. 투자 판단은 본인 책임입니다.",
        "",
        "---",
        "",
    ]

    TABLE_HEADER = [
        "| 티커 | 종목명 | 종합신호 | 현재가 | 등락률 | RSI | ADX | 모멘텀 | 매수/매도 | 수익률Z | 변동폭Z | 거래량Z | 체제 |",
        "|------|--------|----------|--------|--------|-----|-----|--------|-----------|---------|---------|---------|------|",
    ]

    # 한국
    lines += ["## 🇰🇷 한국 시장", ""]
    lines += TABLE_HEADER
    for ticker, (name, sig) in kr_results.items():
        if sig:
            lines.append(combined_row(ticker, name, sig))

    # 미국
    lines += ["", "---", "", "## 🇺🇸 미국 시장", ""]
    lines += TABLE_HEADER
    for ticker, (name, sig) in us_results.items():
        if sig:
            lines.append(combined_row(ticker, name, sig))

    # 용어 설명
    lines += [
        "",
        "---",
        "",
        "## 📖 지표 설명",
        "",
        "### 종합신호",
        "7개 지표 중 매수/매도 신호가 몇 개인지 집계한 결과예요.",
        "",
        "| 신호 | 의미 |",
        "|------|------|",
        "| 🟢 강한매수 | 7개 중 5개 이상 매수 신호 → 강하게 상승 가능성 |",
        "| 🔵 약한매수 | 7개 중 3~4개 매수 신호 → 상승 우세하나 확신 낮음 |",
        "| ⚪ 중립 | 매수/매도 혼재 → 방향성 불명확, 관망 권장 |",
        "| 🟠 약한매도 | 7개 중 3~4개 매도 신호 → 하락 우세하나 확신 낮음 |",
        "| 🔴 강한매도 | 7개 중 5개 이상 매도 신호 → 강하게 하락 가능성 |",
        "",
        "### 7가지 개별 지표",
        "",
        "| 지표 | 매수 조건 | 매도 조건 | 설명 |",
        "|------|-----------|-----------|------|",
        "| RSI | 35 이하 | 65 이상 | 과매도/과매수 측정. 너무 많이 팔렸으면 반등, 너무 많이 올랐으면 조정 가능성 |",
        "| MACD | 히스토그램 양수 | 히스토그램 음수 | 단기/장기 이동평균 차이. 양수면 단기 상승 모멘텀 |",
        "| ADX방향 | DI+ > DI- | DI+ < DI- | 상승/하락 방향성. ADX 수치가 25 이상이어야 신뢰도 높음 |",
        "| 볼린저 | 하단 25% 이하 | 상단 75% 이상 | 가격이 밴드 어디에 위치하는지. 하단이면 저점 근처 |",
        "| MA크로스 | 20일선 > 60일선 | 20일선 < 60일선 | 골든크로스/데드크로스. 중기 추세 방향 |",
        "| 모멘텀 | 1/3/6개월 평균 +2% 이상 | -2% 이하 | 최근 수익률 추세. 오르는 종목이 계속 오르는 경향 |",
        "| 거래량 | - | - | 평소 대비 1.5배 이상이면 급등 (방향은 별도 판단) |",
        "",
        "### HMM 피처 (Z-score)",
        "과거 1년 평균 대비 현재가 얼마나 이상한지를 나타내요. **0이 보통, ±2 이상이면 비정상적**이에요.",
        "",
        "| 컬럼 | 설명 | 해석 |",
        "|------|------|------|",
        "| 수익률Z | 오늘 수익률이 과거 대비 얼마나 큰지 | +2 이상: 비정상 급등 / -2 이하: 비정상 급락 |",
        "| 변동폭Z | 오늘 고저 변동폭이 과거 대비 얼마나 큰지 | +2 이상: 극단적 변동성 (불안정) |",
        "| 거래량Z | 오늘 거래량 변화가 과거 대비 얼마나 큰지 | +2 이상: 거래량 폭발 (세력 개입 가능) |",
        "",
        "### 시장 체제 (HMM 기반 추정)",
        "",
        "| 체제 | 조건 | 전략 힌트 |",
        "|------|------|-----------|",
        "| 🐂 강세 | 수익률Z > 1, 변동폭 낮음 | 추세 추종 유리 |",
        "| 💥 폭락 | 수익률Z < -1.5 | 손절/현금화 고려 |",
        "| ⚡ 고변동 | 변동폭Z > 1.5 | 레버리지 위험, 관망 |",
        "| 😴 횡보 | 수익률/변동폭 모두 낮음 | 돌파 대기, 박스권 매매 |",
        "| 🌀 노이즈 | 위 조건 해당 없음 | 방향성 불명확 |",
        "",
        "---",
        "",
        "*자동 생성: [quant-pipeline](https://github.com/QAZyunho/quant-pipeline/tree/reports)*",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════
def main():
    log.info(f"🚀 분석 시작: {TODAY}")

    kr_results = {}
    us_results = {}

    # 한국
    log.info("📡 한국 데이터 수집 중...")
    import time
    for ticker, name in WATCHLIST["KR"].items():
        try:
            df = fetch_kr(ticker)
            if df.empty:
                log.warning(f"[KR] {ticker} 데이터 없음")
                continue
            df = add_indicators(df)
            sig = get_signal(df)
            kr_results[ticker] = (name, sig)
            log.info(f"[KR] {ticker} {name}: {sig['overall']}")
            time.sleep(0.5)
        except Exception as e:
            log.error(f"[KR] {ticker} 실패: {e}")
            kr_results[ticker] = (name, None)

    # 미국
    log.info("📡 미국 데이터 수집 중...")
    for ticker, name in WATCHLIST["US"].items():
        try:
            df = fetch_us(ticker)
            if df.empty:
                log.warning(f"[US] {ticker} 데이터 없음")
                continue
            df = add_indicators(df)
            sig = get_signal(df)
            us_results[ticker] = (name, sig)
            log.info(f"[US] {ticker} {name}: {sig['overall']}")
            time.sleep(0.3)
        except Exception as e:
            log.error(f"[US] {ticker} 실패: {e}")
            us_results[ticker] = (name, None)

    # 리포트 생성
    log.info("📝 리포트 생성 중...")
    report = build_report(kr_results, us_results)

    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info(f"✅ 리포트 저장: {REPORT_PATH}")

    # latest.md 심볼릭 업데이트 (항상 최신본)
    latest = REPORTS_DIR / "latest.md"
    latest.write_text(report, encoding="utf-8")

    # JSON 요약 저장 (향후 LLM 페이퍼 트레이딩 봇이 읽을 용도)
    summary = {
        "date": TODAY,
        "kr": {t: {"name": n, **s} if s else {"name": n} for t, (n, s) in kr_results.items()},
        "us": {t: {"name": n, **s} if s else {"name": n} for t, (n, s) in us_results.items()},
    }
    # md 저장
    REPORT_PATH.write_text(report, encoding="utf-8")
    (MD_DIR / "latest.md").write_text(report, encoding="utf-8")
    log.info(f"✅ 리포트 저장: {REPORT_PATH}")

    # json 저장 (git 제외)
    (JSON_DIR / f"{TODAY}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("✅ JSON 저장 완료")

if __name__ == "__main__":
    main()