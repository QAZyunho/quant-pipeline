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
import pytz

import numpy as np
import pandas as pd

# ── 시간대 설정 ──────────────────────────────────────────────
KST = pytz.timezone('Asia/Seoul')

# ── 로깅 (한국 시간 사용) ─────────────────────────────────────
class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, KST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S KST')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(KSTFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
log = logging.getLogger(__name__)

# ── 경로 ─────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
html_DIR   = REPORTS_DIR / "html"
JSON_DIR = REPORTS_DIR / "json"
html_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# 한국 시간 기준으로 TODAY 설정
TODAY = datetime.now(KST).strftime("%Y-%m-%d")
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
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")

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
            rz  = sig.get("returns_z", 0)
            raz = sig.get("range_z", 0)
            vcz = sig.get("vol_change_z", 0)

            if rz > 1.0 and raz < 0.5:   regime = "🐂 강세"
            elif rz < -1.5:               regime = "💥 폭락"
            elif raz > 1.5:               regime = "⚡ 고변동"
            elif abs(rz) < 0.3 and raz < 0.3: regime = "😴 횡보"
            else:                         regime = "🌀 노이즈"

            # 펀더멘털
            per = sig.get("per", 0)
            pbr = sig.get("pbr", 0)
            roe = sig.get("roe", sig.get("div", 0))  # 미국은 ROE, 한국은 DIV
            div = sig.get("div", 0)
            
            # 종합 스코어
            comp_scores = sig.get("comprehensive_scores", {})
            swing_score = comp_scores.get("swing", {}).get("score", 0)
            longterm_score = comp_scores.get("longterm", {}).get("score", 0)
            longterm_grade = comp_scores.get("longterm", {}).get("grade", "F")
            recommendation = comp_scores.get("recommendation", "unknown")

            def fmt_ratio(v, low_good=True):
                if v == 0: return "<span style='color:#555'>-</span>"
                if low_good:
                    color = "#1e88e5" if v < 10 else ("#fb8c00" if v < 20 else "#888")
                else:
                    color = "#e53935" if v > 15 else ("#fb8c00" if v > 8 else "#888")
                return f'<span style="color:{color}">{v}</span>'
            
            def fmt_score(score):
                """스코어를 색깔로 표시 (0-100)"""
                if score >= 80: color = "#22c55e"  # 초록
                elif score >= 60: color = "#3b82f6"  # 파랑
                elif score >= 40: color = "#f59e0b"  # 주황
                elif score >= 20: color = "#ef4444"  # 빨강
                else: color = "#6b7280"  # 회색
                return f'<span style="color:{color};font-weight:bold">{score}</span>'
            
            def fmt_grade(grade):
                """등급을 색깔로 표시"""
                color_map = {
                    "A+": "#22c55e", "A": "#3b82f6",
                    "B+": "#06b6d4", "B": "#8b5cf6",
                    "C+": "#f59e0b", "C": "#fb8c00",
                    "D": "#ef4444", "F": "#6b7280"
                }
                color = color_map.get(grade, "#6b7280")
                return f'<span style="color:{color};font-weight:bold">{grade}</span>'
            
            def fmt_recommendation(rec):
                """추천 등급을 색깔과 아이콘으로 표시"""
                rec_map = {
                    "strong_buy": ("🚀", "#22c55e", "강매수"),
                    "buy": ("📈", "#3b82f6", "매수"),
                    "swing_trade": ("⚡", "#8b5cf6", "스윙"),
                    "watchlist": ("👁️", "#06b6d4", "관심"),
                    "hold": ("🤝", "#f59e0b", "보유"),
                    "avoid": ("🚫", "#ef4444", "회피"),
                }
                icon, color, text = rec_map.get(rec, ("❓", "#6b7280", "미지"))
                return f'<span style="color:{color}">{icon} {text}</span>'

            # 차트 링크 생성
            if ticker.isdigit() and len(ticker) == 6:  # 한국 주식
                chart_link = f"https://finance.naver.com/item/fchart.naver?code={ticker}"
            else:  # 미국 주식
                chart_link = f"https://finance.yahoo.com/chart/{ticker}"
            
            rows += f"""
            <tr>
                <td><b><a href="{chart_link}" target="_blank" style="color:#58a6ff">{ticker}</a></b></td>
                <td>{name}</td>
                <td>{color_signal(sig['overall'])}</td>
                <td>{fmt_recommendation(recommendation)}</td>
                <td>{fmt_score(swing_score)}</td>
                <td>{fmt_score(longterm_score)} {fmt_grade(longterm_grade)}</td>
                <td>{sig['close']:,.0f}</td>
                <td>{color_return(sig['returns_1d'])}</td>
                <td>{color_rsi(sig['rsi'])}</td>
                <td>{color_adx(sig['adx'])}</td>
                <td>{color_mom(sig['mom_score'])}</td>
                <td>{sig['buy_count']}/{sig['sell_count']}</td>
                <td>{fmt_ratio(per)}</td>
                <td>{fmt_ratio(pbr)}</td>
                <td>{fmt_ratio(roe, low_good=False)}</td>
                <td><span style="color:#888">{div}%</span></td>
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
                <th title="스윙+장기 종합 투자 추천">최종추천</th>
                <th title="단기 진입 타이밍 점수 (0-100)">스윙점수</th>
                <th title="장기 투자 가치 점수 (0-100)">장기점수</th>
                <th>현재가</th><th>등락률</th>
                <th title="14일 기준. 35↓과매도 65↑과매수">RSI</th>
                <th title="25 이상이면 추세 존재">ADX</th>
                <th title="1/3/6개월 평균 수익률">모멘텀</th>
                <th title="매수/매도 신호 개수">매수/매도</th>
                <th title="주가수익비율. 낮을수록 저평가">PER</th>
                <th title="주가순자산비율. 1 이하면 저평가">PBR</th>
                <th title="자기자본이익률. 높을수록 수익성 좋음">ROE%</th>
                <th title="배당수익률">배당%</th>
                <th title="수익률 Z-score">수익률Z</th>
                <th title="변동폭 Z-score">변동폭Z</th>
                <th title="거래량 Z-score">거래량Z</th>
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

<h2>📈 주요 종목 차트 (클릭하여 상세 차트 보기)</h2>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 16px;">"""

    # 상위 추천 종목들의 차트 위젯 추가
    top_picks = []
    for ticker, (name, sig) in list(kr_results.items())[:3]:  # 상위 3개
        if sig and sig.get("comprehensive_scores", {}).get("recommendation") in ["strong_buy", "buy"]:
            top_picks.append((ticker, name, "KR"))
    
    for ticker, (name, sig) in list(us_results.items())[:3]:  # 상위 3개  
        if sig and sig.get("comprehensive_scores", {}).get("recommendation") in ["strong_buy", "buy"]:
            top_picks.append((ticker, name, "US"))
    
    for ticker, name, market in top_picks[:6]:  # 최대 6개
        if market == "KR":
            widget_symbol = f"KRX:{ticker}"
            naver_link = f"https://finance.naver.com/item/fchart.naver?code={ticker}"
        else:
            widget_symbol = f"NASDAQ:{ticker}" if ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"] else f"NYSE:{ticker}"
            naver_link = f"https://finance.yahoo.com/chart/{ticker}"
            
        html += f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px;">
    <h4 style="margin:0 0 8px 0; color:#e6edf3;">{ticker} - {name}</h4>
    <div style="position:relative; height:300px;">
        <!-- TradingView 위젯 -->
        <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tv_chart_{ticker}&symbol={widget_symbol}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=0d1117&studies=[]&theme=dark&style=1&timezone=Asia%2FSeoul&withdateranges=1&hideideas=1&hidelogo=1&allow_symbol_change=false&width=100%25&height=300"
                style="width:100%; height:100%; border:none;"></iframe>
    </div>
    <div style="margin-top:8px; text-align:center;">
        <a href="{naver_link}" target="_blank" style="color:#58a6ff; text-decoration:none; font-size:0.9em;">
            📊 상세 차트 보기
        </a>
    </div>
</div>"""

    html += """
</div>

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
  <tr><td><strong>RSI</strong><br/>(상대강도지수)</td><td>35 이하 🔵</td><td>65 이상 🔴</td><td><strong>최근 14일 상승일 vs 하락일 강도</strong><br/>0~100 값. 35 이하면 "너무 팔려서 반등 가능성", 65 이상이면 "너무 올라서 조정 가능성". 과열/침체 구간 판단</td></tr>
  <tr><td><strong>MACD</strong><br/>(이동평균수렴확산)</td><td>히스토그램 양수</td><td>히스토그램 음수</td><td><strong>12일선 - 26일선 vs 9일 평균</strong><br/>단기 추세가 장기 추세보다 강한지 측정. 양수면 "상승 모멘텀 시작", 음수면 "하락 모멘텀"</td></tr>
  <tr><td><strong>ADX방향</strong><br/>(추세강도+방향)</td><td>DI+ &gt; DI-</td><td>DI+ &lt; DI-</td><td><strong>상승력(DI+) vs 하락력(DI-) 비교</strong><br/>DI+가 크면 상승세, DI-가 크면 하락세. 단, ADX 25 이상일 때만 신뢰 (추세 뚜렷할 때)</td></tr>
  <tr><td><strong>볼린저밴드</strong><br/>(변동성 밴드)</td><td>하단 25% 이하</td><td>상단 75% 이상</td><td><strong>20일 평균 ± (표준편차 × 2)</strong><br/>정상 가격 범위 대비 현재 위치. 하단이면 "저점 근처", 상단이면 "고점 근처" 가능성</td></tr>
  <tr><td><strong>MA크로스</strong><br/>(이동평균교차)</td><td>20일선 &gt; 60일선</td><td>20일선 &lt; 60일선</td><td><strong>단기 평균 vs 중기 평균</strong><br/>20일선이 60일선 위로 올라가면 골든크로스(상승), 아래로 내려가면 데드크로스(하락) 신호</td></tr>
  <tr><td><strong>모멘텀</strong><br/>(수익률 지속성)</td><td>평균 +2% 이상 🔴</td><td>평균 -2% 이하 🔵</td><td><strong>1개월/3개월/6개월 수익률 평균</strong><br/>상승하는 종목이 계속 상승하는 경향 활용. +2% 이상이면 강한 상승세 지속 중</td></tr>
  <tr><td><strong>거래량</strong><br/>(관심도/유동성)</td><td colspan="2">평소 대비 1.5배 이상이면 급등 (방향은 별도 판단)</td><td><strong>평균 거래량 vs 현재 거래량</strong><br/>거래량 급증은 "큰 변화 신호". 방향은 다른 지표로 판단 (매수/매도 카운트 제외)</td></tr>
</table>

<h3>Z-score 색상 기준</h3>
<table class="legend-table">
  <tr><th>색상</th><th>범위</th><th>의미</th></tr>
  <tr><td style="color:#e53935">빨강</td><td>+2 이상</td><td>비정상적으로 높음</td></tr>
  <tr><td style="color:#fb8c00">주황</td><td>±1 ~ ±2</td><td>주의 구간</td></tr>
  <tr><td style="color:#888">회색</td><td>±1 이내</td><td>정상 범위</td></tr>
  <tr><td style="color:#1e88e5">파랑</td><td>-2 이하</td><td>비정상적으로 낮음</td></tr>
</table>

<h3>펀더멘털 지표 (밸류에이션/수익성)</h3>
<table class="legend-table">
  <tr><th>지표</th><th>좋은 값</th><th>주의 값</th><th>설명</th></tr>
  <tr><td><strong>PER</strong><br/>(주가수익비율)</td><td style="color:#1e88e5">한국: 10 이하<br/>미국: 15 이하</td><td style="color:#fb8c00">20 이상</td><td><strong>주가 ÷ 주당순이익(EPS)</strong><br/>현재 주가가 1년 수익의 몇 배인지 측정. 10이면 "10년간 벌어야 현재 주가만큼 수익". 낮을수록 저평가 (단, 적자 회사는 의미 없음)</td></tr>
  <tr><td><strong>PBR</strong><br/>(주가순자산비율)</td><td style="color:#1e88e5">1.0 이하</td><td style="color:#fb8c00">3.0 이상</td><td><strong>주가 ÷ 주당순자산(BPS)</strong><br/>회사를 청산할 때 주주가 받을 몫 대비 주가. 1.0 이하면 "자산 가치보다 저렴". 성장주는 높아도 OK, 가치주는 낮을수록 좋음</td></tr>
  <tr><td><strong>ROE</strong><br/>(자기자본이익률)</td><td style="color:#e53935">15% 이상</td><td style="color:#888">5% 이하</td><td><strong>순이익 ÷ 자기자본 × 100</strong><br/>주주 돈으로 얼마나 이익을 냈는지. 15%면 "100원 투자해서 15원 벌었다". 경영진의 능력과 사업 수익성을 보여줌</td></tr>
  <tr><td><strong>ROA</strong><br/>(총자산이익률)</td><td style="color:#e53935">8% 이상</td><td style="color:#888">3% 이하</td><td><strong>순이익 ÷ 총자산 × 100</strong><br/>빌린 돈 포함해 모든 자산으로 얼마나 효율적으로 이익을 냈는지. 자산 대비 수익 창출 능력 측정</td></tr>
  <tr><td><strong>배당수익률</strong></td><td style="color:#e53935">3% 이상</td><td style="color:#888">1% 이하</td><td><strong>연간 배당금 ÷ 주가 × 100</strong><br/>주식을 사면 1년간 받는 배당금이 투자금의 몇 %인지. 은행 이자 대신 배당 받고 싶은 투자자에게 중요</td></tr>
</table>

<h3>성장성/안정성 지표</h3>
<table class="legend-table">
  <tr><th>지표</th><th>좋은 값</th><th>주의 값</th><th>설명</th></tr>
  <tr><td><strong>매출성장률</strong><br/>(YoY)</td><td style="color:#e53935">15% 이상</td><td style="color:#888">마이너스</td><td><strong>(올해매출 - 작년매출) ÷ 작년매출 × 100</strong><br/>회사가 얼마나 빠르게 성장하고 있는지 측정. 15%면 "매출이 1년 만에 15% 늘었다". 지속적 성장 가능성의 핵심 지표</td></tr>
  <tr><td><strong>영업이익성장률</strong><br/>(YoY)</td><td style="color:#e53935">20% 이상</td><td style="color:#888">마이너스</td><td><strong>(올해영업이익 - 작년영업이익) ÷ 작년영업이익 × 100</strong><br/>매출 증가와 함께 실제 수익도 늘고 있는지 확인. 매출은 늘어도 영업이익은 줄어들면 경쟁력 악화 신호</td></tr>
  <tr><td><strong>부채비율</strong></td><td style="color:#1e88e5">50% 이하</td><td style="color:#fb8c00">150% 이상</td><td><strong>부채 ÷ 자기자본 × 100</strong><br/>빌린 돈이 자기 돈의 몇 %인지. 50%면 "자기돈 100원에 빌린돈 50원". 높으면 파산 위험, 낮으면 안전하지만 성장 동력 부족할 수도</td></tr>
  <tr><td><strong>유동비율</strong></td><td style="color:#1e88e5">150% 이상</td><td style="color:#fb8c00">100% 이하</td><td><strong>유동자산 ÷ 유동부채 × 100</strong><br/>1년 내 현금화 가능한 자산으로 1년 내 갚아야 할 빚을 감당할 수 있는지. 150% 이상이면 단기 자금난 걱정 없음</td></tr>
  <tr><td><strong>이자보상배율</strong></td><td style="color:#1e88e5">5배 이상</td><td style="color:#fb8c00">3배 이하</td><td><strong>영업이익 ÷ 이자비용</strong><br/>영업으로 번 돈으로 이자를 몇 번 갚을 수 있는지. 5배면 "이자의 5배를 벌고 있어 안전". 3배 미만이면 이자 부담이 과중</td></tr>
</table>

<h3>종합 스코어링 상세</h3>

<h4>🎯 스윙점수 (0~100점) - 단기 진입 타이밍</h4>
<table class="legend-table">
  <tr><th>점수</th><th>매수신호 개수</th><th>의미</th><th>조치</th></tr>
  <tr><td style="color:#22c55e"><strong>80~100점</strong></td><td>6~7개 매수</td><td>강한 단기 상승 신호</td><td>적극적 진입 고려</td></tr>
  <tr><td style="color:#3b82f6"><strong>60~79점</strong></td><td>4~5개 매수</td><td>양호한 진입 타이밍</td><td>조건부 진입</td></tr>
  <tr><td style="color:#f59e0b"><strong>40~59점</strong></td><td>3개 매수</td><td>중립적, 방향성 애매</td><td>추가 확인 필요</td></tr>
  <tr><td style="color:#ef4444"><strong>20~39점</strong></td><td>1~2개 매수</td><td>약한 신호, 하락 우세</td><td>진입 보류</td></tr>
  <tr><td style="color:#6b7280"><strong>0~19점</strong></td><td>0~1개 매수</td><td>강한 하락 신호</td><td>매도 고려</td></tr>
</table>
<p><small>📊 <strong>계산법</strong>: (매수신호 개수 ÷ 7) × 100<br/>
📋 <strong>7가지 신호</strong>: RSI(35↓), MACD(양수), 볼린저(하단25%↓), ADX방향(DI+>DI-), MA크로스(20>60), 모멘텀(+2%↑), 거래량(1.5배↑)</small></p>

<h4>⭐ 장기점수 (0~100점) - 투자가치 평가</h4>
<table class="legend-table">
  <tr><th>등급</th><th>점수</th><th>투자가치</th><th>예시 조건</th></tr>
  <tr><td style="color:#22c55e"><strong>A+ (85~100)</strong></td><td>85~100점</td><td>최우량 종목</td><td>ROE 15%+ 저PER + 고성장</td></tr>
  <tr><td style="color:#3b82f6"><strong>A (75~84)</strong></td><td>75~84점</td><td>우량 종목</td><td>안정성+수익성 양호</td></tr>
  <tr><td style="color:#06b6d4"><strong>B+ (65~74)</strong></td><td>65~74점</td><td>양호한 종목</td><td>일부 지표 우수</td></tr>
  <tr><td style="color:#8b5cf6"><strong>B (55~64)</strong></td><td>55~64점</td><td>평균 수준</td><td>무난하지만 특별하지 않음</td></tr>
  <tr><td style="color:#f59e0b"><strong>C+ (45~54)</strong></td><td>45~54점</td><td>관심 필요</td><td>일부 지표 부족</td></tr>
  <tr><td style="color:#fb8c00"><strong>C (35~44)</strong></td><td>35~44점</td><td>투자 신중</td><td>여러 지표 부족</td></tr>
  <tr><td style="color:#ef4444"><strong>D (25~34)</strong></td><td>25~34점</td><td>위험 종목</td><td>재무상태 불량</td></tr>
  <tr><td style="color:#6b7280"><strong>F (0~24)</strong></td><td>0~24점</td><td>투자 부적합</td><td>대부분 지표 나쁨</td></tr>
</table>
<p><small>📊 <strong>4개 영역별 25점씩</strong>:<br/>
🏆 <strong>수익성(25점)</strong>: ROE 15%↑(10점) + ROA 8%↑(8점) + 영업이익률 10%↑(7점)<br/>
🛡️ <strong>안정성(25점)</strong>: 부채비율 50%↓(10점) + 유동비율 150%↑(8점) + 이자보상배율 5↑(7점)<br/>
💰 <strong>밸류에이션(25점)</strong>: PER 적정(12점) + PBR 저평가(13점)<br/>
📈 <strong>성장성(25점)</strong>: 매출성장률 15%↑(12점) + 영업이익성장률 20%↑(13점)</small></p>

<h4>🎊 최종추천 - 스윙+장기 종합판단</h4>
<table class="legend-table">
  <tr><th>추천</th><th>조건</th><th>투자 성향</th><th>예시</th></tr>
  <tr><td style="color:#22c55e"><strong>🚀 강매수</strong></td><td>스윙70+ & 장기60+</td><td>적극적 성장 추구</td><td>스윙85 + 장기A</td></tr>
  <tr><td style="color:#3b82f6"><strong>📈 매수</strong></td><td>스윙50+ & 장기50+</td><td>안정적 수익 추구</td><td>스윙65 + 장기B+</td></tr>
  <tr><td style="color:#8b5cf6"><strong>⚡ 스윙</strong></td><td>스윙70+ & 장기30~70</td><td>단기 트레이딩</td><td>스윙80 + 장기C</td></tr>
  <tr><td style="color:#06b6d4"><strong>👁️ 관심</strong></td><td>스윙30~70 & 장기70+</td><td>장기 관점 대기</td><td>스윙40 + 장기A</td></tr>
  <tr><td style="color:#f59e0b"><strong>🤝 보유</strong></td><td>스윙40+ & 장기40+</td><td>현상 유지</td><td>스윙45 + 장기B</td></tr>
  <tr><td style="color:#ef4444"><strong>🚫 회피</strong></td><td>장기30 미만</td><td>투자 부적합</td><td>장기 F등급</td></tr>
</table>
<p><small>💡 <strong>활용 팁</strong>: 개인 투자 성향에 맞춰 선택적 참고. 스윙 점수는 진입 타이밍, 장기 점수는 종목 선별 기준으로 활용</small></p>

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
    from fundamental import enrich_with_fundamental, calculate_comprehensive_scores
    for ticker, name in WATCHLIST["KR"].items():
        try:
            df = fetch_kr(ticker)
            if df.empty:
                log.warning(f"[KR] {ticker} 데이터 없음")
                continue
            df = add_indicators(df)
            sig = get_signal(df)
            # 펀더멘털 + 목표주가 추가
            fund = enrich_with_fundamental(ticker, "KR", sig["close"])
            sig.update(fund)
            
            # 종합 스코어 계산
            comprehensive = calculate_comprehensive_scores(ticker, "KR", sig, fund)
            sig["comprehensive_scores"] = comprehensive
            
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
            # 펀더멘털 추가
            fund = enrich_with_fundamental(ticker, "US", sig["close"])
            sig.update(fund)
            
            # 종합 스코어 계산
            comprehensive = calculate_comprehensive_scores(ticker, "US", sig, fund)
            sig["comprehensive_scores"] = comprehensive
            
            us_results[ticker] = (name, sig)
            log.info(f"[US] {ticker} {name}: {sig['overall']}")
            time.sleep(0.3)
        except Exception as e:
            log.error(f"[US] {ticker} 실패: {e}")
            us_results[ticker] = (name, None)

    # 리포트 생성
    log.info("📝 리포트 생성 중...")
    report = build_report(kr_results, us_results)

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