"""
펀더멘털 수집 모듈
================================
한국: pykrx (PER/PBR/EPS/DIV) + OpenDartReader (ROE/ROA/재무제표)
미국: yfinance (PER/PBR/ROE/ROA)

사용법:
    from fundamental import enrich_with_fundamental
"""

import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 한국: pykrx 펀더멘털
# ════════════════════════════════════════════════════════════
def get_kr_fundamental(ticker: str) -> dict:
    try:
        from pykrx import stock
        today = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=14)).strftime("%Y%m%d")

        try:
            df = stock.get_market_fundamental(start, today, ticker)
        except Exception:
            return {}  # KRX 서버 오류 (주말 등) → 조용히 빈 값 반환

        if df is None or df.empty:
            return {}

        row = df.iloc[-1]
        return {
            "per": round(float(row.get("PER", 0)), 2),
            "pbr": round(float(row.get("PBR", 0)), 2),
            "eps": round(float(row.get("EPS", 0)), 0),
            "bps": round(float(row.get("BPS", 0)), 0),
            "div": round(float(row.get("DIV", 0)), 2),
        }
    except Exception as e:
        log.warning(f"[KR] {ticker} 펀더멘털 수집 실패: {e}")
        return {}

# ════════════════════════════════════════════════════════════
# 미국: yfinance 펀더멘털
# ════════════════════════════════════════════════════════════
def get_us_fundamental(ticker: str) -> dict:
    """yfinance로 PER, PBR, ROE, ROA, 부채비율, 배당수익률 수집."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "per":            round(info.get("trailingPE")      or 0, 2),
            "forward_per":    round(info.get("forwardPE")       or 0, 2),
            "pbr":            round(info.get("priceToBook")     or 0, 2),
            "roe":            round((info.get("returnOnEquity") or 0) * 100, 2),
            "roa":            round((info.get("returnOnAssets") or 0) * 100, 2),
            "debt_equity":    round(info.get("debtToEquity")    or 0, 2),
            "div":            round((info.get("dividendYield")  or 0) * 100, 2),
            "revenue_growth": round((info.get("revenueGrowth")  or 0) * 100, 2),
            "earnings_growth":round((info.get("earningsGrowth") or 0) * 100, 2),
        }
    except Exception as e:
        log.warning(f"[US] {ticker} 펀더멘털 수집 실패: {e}")
        return {}


# ════════════════════════════════════════════════════════════
# 통합 수집 함수
# ════════════════════════════════════════════════════════════
def enrich_with_fundamental(ticker: str, market: str, current_price: float = 0) -> dict:
    result = {}

    if market == "KR":
        fund = get_kr_fundamental(ticker)
        result.update(fund)

        try:
            from dart_collector import get_dart_financials
            dart_data = get_dart_financials(ticker)
            if dart_data:
                result["roe"]        = dart_data.get("roe", 0)
                result["roa"]        = dart_data.get("roa", 0)
                result["debt_ratio"] = dart_data.get("debt_ratio", 0)
                result["op_margin"]  = dart_data.get("op_margin", 0)
                result["dart"]       = dart_data
        except Exception as e:
            log.warning(f"[DART] {ticker} 수집 실패: {e}")

    elif market == "US":
        fund = get_us_fundamental(ticker)
        result.update(fund)

    return result
