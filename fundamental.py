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


# ════════════════════════════════════════════════════════════
# 스윙/장기 분리 스코어링 시스템
# ════════════════════════════════════════════════════════════

def calculate_swing_score(indicators_summary: dict) -> dict:
    """
    스윙 트레이딩용 기술적 지표 점수 (0-100)
    기존 지표들로 단기 진입 타이밍 판단
    """
    if not indicators_summary:
        return {"score": 0, "signal": "neutral", "details": {}}
    
    total_signals = 7  # RSI, MACD, BB, ADX, 볼륨, 모멘텀, 수익률Z
    buy_count = indicators_summary.get("buy_count", 0)
    sell_count = indicators_summary.get("sell_count", 0)
    
    # 0-100 점수로 변환
    score = round((buy_count / total_signals) * 100)
    
    # 시그널 결정
    if buy_count >= 5:
        signal = "strong_buy"
    elif buy_count >= 3:
        signal = "buy"
    elif sell_count >= 3:
        signal = "sell"
    elif sell_count >= 5:
        signal = "strong_sell"
    else:
        signal = "neutral"
    
    return {
        "score": score,
        "signal": signal,
        "buy_signals": buy_count,
        "total_signals": total_signals,
        "details": {
            "rsi": indicators_summary.get("rsi_signal", "neutral"),
            "macd": indicators_summary.get("macd_signal", "neutral"),
            "bollinger": indicators_summary.get("bb_signal", "neutral"),
            "adx": indicators_summary.get("adx_signal", "neutral"),
            "volume": indicators_summary.get("volume_signal", "neutral"),
            "momentum": indicators_summary.get("momentum_signal", "neutral"),
            "returns": indicators_summary.get("returns_signal", "neutral"),
        }
    }


def calculate_longterm_score(fundamental_data: dict, market: str = "KR") -> dict:
    """
    장기 투자용 펀더멘털 점수 (0-100)
    수익성, 안정성, 밸류에이션, 성장성 종합 평가
    """
    if not fundamental_data:
        return {"score": 0, "grade": "F", "details": {}}
    
    scores = {}
    dart_data = fundamental_data.get("dart", {})
    
    # ─── 1. 수익성 점수 (25점) ───
    roe = fundamental_data.get("roe", 0) or dart_data.get("roe", 0)
    roa = fundamental_data.get("roa", 0) or dart_data.get("roa", 0)
    op_margin = dart_data.get("op_margin", 0)
    
    profit_score = 0
    if roe >= 15: profit_score += 10
    elif roe >= 10: profit_score += 7
    elif roe >= 5: profit_score += 4
    
    if roa >= 8: profit_score += 8
    elif roa >= 5: profit_score += 5
    elif roa >= 3: profit_score += 2
    
    if op_margin >= 10: profit_score += 7
    elif op_margin >= 5: profit_score += 4
    elif op_margin >= 3: profit_score += 2
    
    scores["profitability"] = min(profit_score, 25)
    
    # ─── 2. 안정성 점수 (25점) ───
    debt_ratio = fundamental_data.get("debt_ratio", 0) or dart_data.get("debt_ratio", 0)
    current_ratio = dart_data.get("current_ratio", 0)
    interest_coverage = dart_data.get("interest_coverage", 0)
    
    stability_score = 0
    if debt_ratio <= 50: stability_score += 10
    elif debt_ratio <= 100: stability_score += 6
    elif debt_ratio <= 150: stability_score += 3
    
    if current_ratio >= 150: stability_score += 8
    elif current_ratio >= 120: stability_score += 5
    elif current_ratio >= 100: stability_score += 2
    
    if interest_coverage >= 5: stability_score += 7
    elif interest_coverage >= 3: stability_score += 4
    elif interest_coverage >= 1.5: stability_score += 2
    
    scores["stability"] = min(stability_score, 25)
    
    # ─── 3. 밸류에이션 점수 (25점) ───
    per = fundamental_data.get("per", 0)
    pbr = fundamental_data.get("pbr", 0)
    
    valuation_score = 0
    if market == "KR":
        if 0 < per <= 10: valuation_score += 12
        elif 10 < per <= 15: valuation_score += 8
        elif 15 < per <= 20: valuation_score += 5
        
        if 0 < pbr <= 1: valuation_score += 13
        elif 1 < pbr <= 1.5: valuation_score += 9
        elif 1.5 < pbr <= 2: valuation_score += 5
    else:  # US
        if 0 < per <= 15: valuation_score += 12
        elif 15 < per <= 25: valuation_score += 8
        elif 25 < per <= 35: valuation_score += 5
        
        if 0 < pbr <= 2: valuation_score += 13
        elif 2 < pbr <= 3: valuation_score += 9
        elif 3 < pbr <= 4: valuation_score += 5
    
    scores["valuation"] = min(valuation_score, 25)
    
    # ─── 4. 성장성 점수 (25점) ───
    revenue_growth = dart_data.get("revenue_growth", 0) or fundamental_data.get("revenue_growth", 0)
    operating_growth = dart_data.get("operating_growth", 0) or fundamental_data.get("earnings_growth", 0)
    
    growth_score = 0
    if revenue_growth >= 15: growth_score += 12
    elif revenue_growth >= 8: growth_score += 8
    elif revenue_growth >= 3: growth_score += 4
    elif revenue_growth >= 0: growth_score += 1
    
    if operating_growth >= 20: growth_score += 13
    elif operating_growth >= 10: growth_score += 8
    elif operating_growth >= 5: growth_score += 4
    elif operating_growth >= 0: growth_score += 1
    
    scores["growth"] = min(growth_score, 25)
    
    # ─── 총점 계산 ───
    total_score = sum(scores.values())
    
    # 등급 계산
    if total_score >= 85: grade = "A+"
    elif total_score >= 75: grade = "A"
    elif total_score >= 65: grade = "B+"
    elif total_score >= 55: grade = "B"
    elif total_score >= 45: grade = "C+"
    elif total_score >= 35: grade = "C"
    elif total_score >= 25: grade = "D"
    else: grade = "F"
    
    return {
        "score": total_score,
        "grade": grade,
        "details": scores,
        "explanation": {
            "profitability": f"ROE {roe:.1f}%, ROA {roa:.1f}%, 영업이익률 {op_margin:.1f}%",
            "stability": f"부채비율 {debt_ratio:.1f}%, 유동비율 {current_ratio:.1f}%, 이자보상배율 {interest_coverage:.1f}",
            "valuation": f"PER {per:.1f}, PBR {pbr:.1f}",
            "growth": f"매출성장률 {revenue_growth:.1f}%, 영업이익성장률 {operating_growth:.1f}%",
        }
    }


def calculate_comprehensive_scores(ticker: str, market: str, technical_summary: dict, fundamental_data: dict) -> dict:
    """
    스윙 + 장기 종합 스코어 계산
    """
    swing = calculate_swing_score(technical_summary)
    longterm = calculate_longterm_score(fundamental_data, market)
    
    return {
        "ticker": ticker,
        "market": market,
        "swing": swing,
        "longterm": longterm,
        "recommendation": _get_recommendation(swing, longterm),
    }


def _get_recommendation(swing: dict, longterm: dict) -> str:
    """
    스윙 + 장기 점수 기반 최종 투자 추천
    """
    swing_score = swing.get("score", 0)
    longterm_score = longterm.get("score", 0)
    
    # 장기 점수가 너무 낮으면 스윙이 좋아도 비추천
    if longterm_score < 30:
        return "avoid"
    
    # 스윙 강한 매수 + 장기 양호 → 강한 매수
    if swing_score >= 70 and longterm_score >= 60:
        return "strong_buy"
    
    # 스윙 매수 + 장기 양호 → 매수
    if swing_score >= 50 and longterm_score >= 50:
        return "buy"
    
    # 장기만 좋음 → 관심종목
    if longterm_score >= 70:
        return "watchlist"
    
    # 스윙만 좋음 → 단기매매
    if swing_score >= 70:
        return "swing_trade"
    
    # 둘 다 보통 → 중립
    if swing_score >= 40 and longterm_score >= 40:
        return "hold"
    
    return "avoid"
