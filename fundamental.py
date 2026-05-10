"""
펀더멘털 + 목표주가 수집 모듈
================================
한국: pykrx (PER/PBR/ROE) + 네이버 증권 스크래핑 (목표주가)
미국: yfinance (PER/PBR/ROE)

사용법:
    from fundamental import get_kr_fundamental, get_us_fundamental, get_target_price
"""

import time
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://finance.naver.com",
}


# ════════════════════════════════════════════════════════════
# 한국: pykrx 펀더멘털
# ════════════════════════════════════════════════════════════
def get_kr_fundamental(ticker: str) -> dict:
    """
    pykrx로 한국 종목 펀더멘털 수집.
    PER, PBR, EPS, BPS, DIV(배당수익률) 반환.
    """
    try:
        from pykrx import stock
        today = datetime.today().strftime("%Y%m%d")
        df = stock.get_market_fundamental(today, today, ticker)

        if df.empty:
            # 오늘 데이터 없으면 최근 5영업일 시도
            from datetime import timedelta
            start = (datetime.today() - timedelta(days=7)).strftime("%Y%m%d")
            df = stock.get_market_fundamental(start, today, ticker)

        if df.empty:
            return {}

        row = df.iloc[-1]
        return {
            "per":  round(float(row.get("PER", 0)), 2),
            "pbr":  round(float(row.get("PBR", 0)), 2),
            "eps":  round(float(row.get("EPS", 0)), 0),
            "bps":  round(float(row.get("BPS", 0)), 0),
            "div":  round(float(row.get("DIV", 0)), 2),  # 배당수익률 %
        }
    except Exception as e:
        log.warning(f"[KR] {ticker} 펀더멘털 수집 실패: {e}")
        return {}


# ════════════════════════════════════════════════════════════
# 미국: yfinance 펀더멘털
# ════════════════════════════════════════════════════════════
def get_us_fundamental(ticker: str) -> dict:
    """
    yfinance로 미국 종목 펀더멘털 수집.
    PER, PBR, ROE, ROA, 부채비율, 배당수익률 반환.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "per":           round(info.get("trailingPE")   or 0, 2),
            "forward_per":   round(info.get("forwardPE")    or 0, 2),
            "pbr":           round(info.get("priceToBook")  or 0, 2),
            "roe":           round((info.get("returnOnEquity")  or 0) * 100, 2),
            "roa":           round((info.get("returnOnAssets")  or 0) * 100, 2),
            "debt_equity":   round(info.get("debtToEquity") or 0, 2),
            "div":           round((info.get("dividendYield") or 0) * 100, 2),
            "revenue_growth":round((info.get("revenueGrowth") or 0) * 100, 2),
            "earnings_growth":round((info.get("earningsGrowth") or 0) * 100, 2),
        }
    except Exception as e:
        log.warning(f"[US] {ticker} 펀더멘털 수집 실패: {e}")
        return {}


# ════════════════════════════════════════════════════════════
# 네이버 증권: 목표주가 스크래핑
# ════════════════════════════════════════════════════════════
def get_target_price(ticker: str) -> dict:
    """
    네이버 증권에서 증권사 목표주가 컨센서스 수집.

    반환:
        {
            "target_avg":   평균 목표주가,
            "target_high":  최고 목표주가,
            "target_low":   최저 목표주가,
            "analyst_count": 분석가 수,
            "upside":       현재가 대비 상승여력 (%),
            "details": [{"firm": 증권사, "target": 목표주가, "opinion": 의견}, ...]
        }
    """
    try:
        url = f"https://finance.naver.com/item/consensus.naver?code={ticker}"
        resp = requests.get(url, headers=HEADERS, timeout=10)

        if resp.status_code != 200:
            log.warning(f"[목표주가] {ticker} 네이버 응답 {resp.status_code}")
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")

        # 컨센서스 테이블 파싱
        table = soup.find("table", {"class": "type1"})
        if not table:
            # 대안 테이블 탐색
            tables = soup.find_all("table")
            table = tables[0] if tables else None

        if not table:
            return {}

        rows = table.find_all("tr")
        details = []
        prices = []

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            texts = [c.get_text(strip=True) for c in cols]
            # 증권사명 / 투자의견 / 목표주가 형태
            try:
                firm    = texts[0]
                opinion = texts[1]
                price_str = texts[2].replace(",", "").replace("원", "").strip()
                if price_str.isdigit():
                    price = int(price_str)
                    prices.append(price)
                    details.append({
                        "firm":    firm,
                        "opinion": opinion,
                        "target":  price,
                    })
            except Exception:
                continue

        if not prices:
            return {}

        # 현재가 가져오기 (상승여력 계산용)
        current_price = _get_current_price_naver(ticker)
        upside = 0.0
        if current_price and current_price > 0:
            avg = sum(prices) / len(prices)
            upside = round((avg / current_price - 1) * 100, 2)

        return {
            "target_avg":    round(sum(prices) / len(prices), 0),
            "target_high":   max(prices),
            "target_low":    min(prices),
            "analyst_count": len(prices),
            "upside":        upside,
            "details":       details[:5],  # 최근 5개만
        }

    except Exception as e:
        log.warning(f"[목표주가] {ticker} 수집 실패: {e}")
        return {}


def _get_current_price_naver(ticker: str) -> float:
    """네이버 증권에서 현재가 조회 (상승여력 계산용)."""
    try:
        url = f"https://finance.naver.com/item/main.naver?code={ticker}"
        resp = requests.get(url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        price_tag = soup.find("strong", {"id": "chart_area"})
        if not price_tag:
            price_tag = soup.find("p", {"class": "no_today"})
        if price_tag:
            txt = price_tag.get_text(strip=True).replace(",", "")
            return float(txt) if txt.isdigit() else 0.0
    except Exception:
        pass
    return 0.0


# ════════════════════════════════════════════════════════════
# 통합 수집 함수
# ════════════════════════════════════════════════════════════
def enrich_with_fundamental(ticker, market, current_price=0):
    result = {}

    if market == "KR":
        # pykrx: PER, PBR, DIV
        fund = get_kr_fundamental(ticker)
        result.update(fund)

        # OpenDart: ROE, ROA, 재무제표
        from dart_collector import get_dart_financials
        dart_data = get_dart_financials(ticker)
        if dart_data:
            result["roe"]        = dart_data.get("roe", 0)
            result["roa"]        = dart_data.get("roa", 0)
            result["debt_ratio"] = dart_data.get("debt_ratio", 0)
            result["op_margin"]  = dart_data.get("op_margin", 0)
            result["dart"]       = dart_data  # 전체 재무데이터 보관

    elif market == "US":
        fund = get_us_fundamental(ticker)
        result.update(fund)

    return result
