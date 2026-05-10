"""
일별 자동 분석 스크립트
========================
GitHub Actions에서 매일 실행되어 분석 리포트를 생성합니다.
결과물: reports/YYYY-MM-DD.html
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
html_DIR   = REPORTS_DIR / "html"
JSON_DIR = REPORTS_DIR / "json"
html_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.today().strftime("%Y-%m-%d")
REPORT_PATH = html_DIR / f"{TODAY}.html"
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

    def color_return(val: float) -> str:
        color = "#e53935" if val > 0 else "#1e88e5"
        return f'<span style="color:{color}">{val:+.2f}%</span>'

    def color_rsi(val: float) -> str:
        if val > 65:
            color = "#e53935"
        elif val < 35:
            color = "#1e88e5"
        else:
            color = "#888"
        return f'<span style="color:{color}">{val}</span>'

    def color_adx(val: float) -> str:
        color = "#e53935" if val >= 25 else "#888"
        return f'<span style="color:{color}">{val}</span>'

    def color_mom(val: float) -> str:
        color = "#e53935" if val > 2 else ("#1e88e5" if val < -2 else "#888")
        return f'<span style="color:{color}">{val:+.2f}%</span>'

    def color_z(val: float) -> str:
        if val > 2:
            color = "#e53935"
        elif val < -2:
            color = "#1e88e5"
        elif abs(val) > 1:
            color = "#fb8c00"
        else:
            color = "#888"
        return f'<span style="color:{color}">{val:+.2f}</span>'

    def color_signal(overall: str) -> str:
        colors = {
            "🟢 강한매수": "#e53935",
            "🔵 약한매수": "#1e88e5",
            "⚪ 중립":     "#888",
            "🟠 약한매도": "#fb8c00",
            "🔴 강한매도": "#5e35b1",
        }
        color = colors.get(overall, "#888")
        return f'<span style="color:{color};font-weight:bold">{overall}</span>'

    def make_table(results: dict) -> str:
        rows = ""
        for ticker, (name, sig) in results.items():
            if not sig:
                continue
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

            rows += f"""
            <tr>
                <td><b>{ticker}</b></td>
                <td>{name}</td>
                <td>{color_signal(sig['overall'])}</td>
                <td>{sig['close']:,.0f}</td>
                <td>{color_return(sig['returns_1d'])}</td>
                <td>{color_rsi(sig['rsi'])}</td>
                <td>{color_adx(sig['adx'])}</td>
                <td>{color_mom(sig['mom_score'])}</td>
                <td>{sig['buy_count']}/{sig['sell_count']}</td>
                <td>{color_z(rz)}</td>
                <td>{color_z(raz)}</td>
                <td>{color_z(vcz)}</td>
                <td>{regime}</td>
            </tr>"""
        return rows

    TABLE_HEADER = """
    <table>
        <thead>
            <tr>
                <th>티커</th><th>종목명</th><th>종합신호</th>
                <th>현재가</th><th>등락률</th>
                <th title="14일 기준. 35↓과매도(파랑) 65↑과매수(빨강)">RSI</th>
                <th title="25 이상이면 추세 존재(빨강)">ADX</th>
                <th title="1/3/6개월 평균 수익률">모멘텀</th>
                <th title="7개 지표 중 매수/매도 신호 개수">매수/매도</th>
                <th title="수익률 Z-score. ±2 이상이면 비정상">수익률Z</th>
                <th title="변동폭 Z-score. +2 이상이면 극단적 변동">변동폭Z</th>
                <th title="거래량 Z-score. +2 이상이면 거래량 폭발">거래량Z</th>
                <th>체제</th>
            </tr>
        </thead>
        <tbody>
    """

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>퀀트 리포트 {TODAY}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1   {{ color: #e6edf3; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  h2   {{ color: #e6edf3; margin-top: 32px; }}
  h3   {{ color: #8b949e; }}
  .meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 24px; }}
  .warn {{ background: #2d1b00; border-left: 4px solid #fb8c00;
           padding: 8px 16px; border-radius: 4px; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px;
           font-size: 0.9em; }}
  th    {{ background: #161b22; color: #8b949e; padding: 8px 12px;
           text-align: left; border-bottom: 1px solid #30363d;
           cursor: help; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #161b22; }}
  details {{ margin-top: 32px; }}
  summary {{ cursor: pointer; color: #58a6ff; font-size: 1.1em;
             font-weight: bold; padding: 8px 0; }}
  .legend-table td, .legend-table th {{
    font-size: 0.85em; padding: 5px 10px; }}
  .footer {{ color: #8b949e; font-size: 0.8em; margin-top: 40px;
             border-top: 1px solid #30363d; padding-top: 12px; }}
</style>
</head>
<body>

<h1>📊 퀀트 일별 분석 리포트</h1>
<div class="meta">생성일시: {now}</div>
<div class="warn">⚠️ 본 리포트는 <b>보조 참고용</b>입니다. 투자 판단은 본인 책임입니다.</div>

<h2>🇰🇷 한국 시장</h2>
{TABLE_HEADER}
{make_table(kr_results)}
        </tbody>
    </table>

<h2>🇺🇸 미국 시장</h2>
{TABLE_HEADER}
{make_table(us_results)}
        </tbody>
    </table>

<details>
<summary>📖 지표 설명 (클릭해서 펼치기)</summary>

<h3>종합신호</h3>
<table class="legend-table">
  <tr><th>신호</th><th>의미</th></tr>
  <tr><td>🟢 강한매수</td><td>7개 중 5개 이상 매수 → 상승 가능성 높음</td></tr>
  <tr><td>🔵 약한매수</td><td>7개 중 3~4개 매수 → 상승 우세하나 확신 낮음</td></tr>
  <tr><td>⚪ 중립</td><td>매수/매도 혼재 → 방향성 불명확, 관망 권장</td></tr>
  <tr><td>🟠 약한매도</td><td>7개 중 3~4개 매도 → 하락 우세하나 확신 낮음</td></tr>
  <tr><td>🔴 강한매도</td><td>7개 중 5개 이상 매도 → 하락 가능성 높음</td></tr>
</table>

<h3>7가지 개별 지표</h3>
<table class="legend-table">
  <tr><th>지표</th><th>매수 조건</th><th>매도 조건</th><th>설명</th></tr>
  <tr><td>RSI</td><td>35 이하 🔵</td><td>65 이상 🔴</td><td>과매도/과매수 측정. 너무 팔렸으면 반등, 너무 올랐으면 조정 가능성</td></tr>
  <tr><td>MACD</td><td>히스토그램 양수</td><td>히스토그램 음수</td><td>단기/장기 이동평균 차이. 양수면 단기 상승 모멘텀</td></tr>
  <tr><td>ADX방향</td><td>DI+ &gt; DI-</td><td>DI+ &lt; DI-</td><td>ADX 25🔴 이상이어야 신뢰도 높음</td></tr>
  <tr><td>볼린저</td><td>하단 25% 이하</td><td>상단 75% 이상</td><td>가격이 밴드 어디 위치하는지. 하단이면 저점 근처</td></tr>
  <tr><td>MA크로스</td><td>20일선 &gt; 60일선</td><td>20일선 &lt; 60일선</td><td>골든/데드크로스. 중기 추세 방향</td></tr>
  <tr><td>모멘텀</td><td>평균 +2% 이상 🔴</td><td>평균 -2% 이하 🔵</td><td>1/3/6개월 평균 수익률. 오르는 종목이 계속 오르는 경향</td></tr>
  <tr><td>거래량</td><td colspan="2">평소 대비 1.5배 이상이면 급등 (방향은 별도 판단)</td><td>매수/매도 카운트 미포함</td></tr>
</table>

<h3>Z-score 색상 기준</h3>
<table class="legend-table">
  <tr><th>색상</th><th>범위</th><th>의미</th></tr>
  <tr><td style="color:#e53935">빨강</td><td>+2 이상</td><td>비정상적으로 높음</td></tr>
  <tr><td style="color:#fb8c00">주황</td><td>±1 ~ ±2</td><td>주의 구간</td></tr>
  <tr><td style="color:#888">회색</td><td>±1 이내</td><td>정상 범위</td></tr>
  <tr><td style="color:#1e88e5">파랑</td><td>-2 이하</td><td>비정상적으로 낮음</td></tr>
</table>

<h3>시장 체제</h3>
<table class="legend-table">
  <tr><th>체제</th><th>조건</th><th>전략 힌트</th></tr>
  <tr><td>🐂 강세</td><td>수익률Z &gt; 1, 변동폭 낮음</td><td>추세 추종 유리</td></tr>
  <tr><td>💥 폭락</td><td>수익률Z &lt; -1.5</td><td>손절/현금화 고려</td></tr>
  <tr><td>⚡ 고변동</td><td>변동폭Z &gt; 1.5</td><td>레버리지 위험, 관망</td></tr>
  <tr><td>😴 횡보</td><td>수익률/변동폭 모두 낮음</td><td>돌파 대기, 박스권 매매</td></tr>
  <tr><td>🌀 노이즈</td><td>위 조건 해당 없음</td><td>방향성 불명확</td></tr>
</table>

</details>

<div class="footer">
  자동 생성: <a href="https://github.com/QAZyunho/quant-pipeline/tree/reports" style="color:#58a6ff">quant-pipeline</a>
</div>

</body>
</html>"""

    return html

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

    # latest.html 심볼릭 업데이트 (항상 최신본)
    latest = REPORTS_DIR / "latest.html"
    latest.write_text(report, encoding="utf-8")

    # JSON 요약 저장 (향후 LLM 페이퍼 트레이딩 봇이 읽을 용도)
    summary = {
        "date": TODAY,
        "kr": {t: {"name": n, **s} if s else {"name": n} for t, (n, s) in kr_results.items()},
        "us": {t: {"name": n, **s} if s else {"name": n} for t, (n, s) in us_results.items()},
    }
    # html 저장
    REPORT_PATH.write_text(report, encoding="utf-8")
    (html_DIR / "latest.html").write_text(report, encoding="utf-8")
    log.info(f"✅ 리포트 저장: {REPORT_PATH}")

    # json 저장 (git 제외)
    (JSON_DIR / f"{TODAY}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("✅ JSON 저장 완료")

if __name__ == "__main__":
    main()