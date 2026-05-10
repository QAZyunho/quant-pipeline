"""
OpenDartReader 재무제표 수집 모듈
===================================
DART 공시 기반 한국 상장사 재무데이터 수집.

가져오는 데이터:
  - 손익계산서: 매출액, 영업이익, 당기순이익, 영업이익률
  - 재무상태표: 자산, 부채, 자본
  - 핵심 지표: ROE, ROA, 부채비율, 영업이익률

API 키 발급:
  https://opendart.fss.or.kr → 인증키 신청
  GitHub Secrets: DART_API_KEY
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# 캐시 경로
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 보고서 코드
REPORT_CODES = {
    "annual": "11011",  # 사업보고서 (연간)
    "q1":     "11013",  # 1분기보고서
    "q2":     "11012",  # 반기보고서
    "q3":     "11014",  # 3분기보고서
}

# 우선주 → 보통주 매핑
PREFERRED_MAP = {
    "000155": "000150",  # 두산우 → 두산
    "005935": "005930",  # 삼성전자우 → 삼성전자
}

# 재무제표 계정명 매핑
ACCOUNT_MAP = {
    "매출액":                "revenue",
    "수익(매출액)":           "revenue",
    "영업이익":               "operating_income",
    "영업이익(손실)":         "operating_income",
    "당기순이익":             "net_income",
    "당기순이익(손실)":       "net_income",
    "법인세비용차감전순이익":  "pretax_income",
    "자산총계":               "total_assets",
    "부채총계":               "total_liabilities",
    "자본총계":               "total_equity",
    "자본금":                 "capital",
    "유동자산":               "current_assets",
    "유동부채":               "current_liabilities",
    "이자비용":               "interest_expense",
    "금융비용":               "interest_expense",
}


def get_dart_financials(
    ticker: str,
    api_key: Optional[str] = None,
    year: Optional[int] = None,
    include_previous: bool = True,
) -> dict:
    """
    OpenDartReader로 재무제표 핵심 지표 수집.
    7일 캐시 적용 — 재무제표는 분기별 데이터라 매일 바뀌지 않음.
    
    Args:
        include_previous: True면 전년도 데이터도 함께 수집 (성장률 계산용)
    """
    # ── 우선주 → 보통주 변환 ──────────────────────────────────
    dart_ticker = PREFERRED_MAP.get(ticker, ticker)
    
    # ── 캐시 확인 ────────────────────────────────────────────
    cache_path = CACHE_DIR / f"dart_{dart_ticker}.json"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 7 * 86400:  # 7일 이내
            log.debug(f"[DART] {ticker} 캐시 사용")
            return json.loads(cache_path.read_text(encoding="utf-8"))

    # ── API 키 확인 ───────────────────────────────────────────
    key = api_key or os.environ.get("DART_API_KEY", "")
    if not key:
        log.warning(f"[DART] API 키 없음 → DART_API_KEY 환경변수 설정 필요")
        return {}

    # ── OpenDartReader 초기화 ─────────────────────────────────
    try:
        import OpenDartReader as odr
        dart = odr(key)
    except Exception as e:
        log.error(f"[DART] OpenDartReader 초기화 실패: {e}")
        return {}

    # ── 연도 결정 ─────────────────────────────────────────────
    if year is None:
        now = datetime.today()
        year = now.year - 1 if now.month >= 4 else now.year - 2

    # 최종 결과
    result = {}

    # ── 현재 연도 데이터 수집 ──────────────────────────────────
    current_data = _get_year_data(dart, dart_ticker, year)
    if current_data:
        result.update(current_data)
        result["year"] = year

    # ── 전년도 데이터 수집 (성장률 계산용) ─────────────────────
    if include_previous and current_data:
        prev_year = year - 1
        prev_data = _get_year_data(dart, dart_ticker, prev_year)
        if prev_data:
            result["previous_year"] = prev_year
            result["previous"] = prev_data
            result = _calc_growth_rates(result)

    if result:
        # 캐시 저장
        cache_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return result

    log.warning(f"[DART] {ticker} 재무제표 수집 실패")
    
    # 우선주인 경우 로그에 변환 정보 추가
    if ticker in PREFERRED_MAP:
        log.info(f"[DART] {ticker} → {dart_ticker} 변환하여 조회했으나 실패")
    return {}


def _get_year_data(dart, ticker: str, year: int) -> dict:
    """특정 연도의 재무제표 데이터 수집."""
    attempts = [
        ("연간",      "11011", "CFS"),
        ("반기",      "11012", "CFS"),
        ("1분기",     "11013", "CFS"),
        ("연간(별도)", "11011", "OFS"),
        ("반기(별도)", "11012", "OFS"),
    ]

    for report_type, code, fs_div in attempts:
        try:
            df = dart.finstate_all(ticker, year, reprt_code=code, fs_div=fs_div)
            if df is not None and not df.empty:
                log.info(f"[DART] {ticker} {year}년 {report_type} 수집 완료")
                result = _parse_finstate(df)
                result["report_type"] = report_type
                result = _calc_ratios(result)
                return result
        except Exception as e:
            log.debug(f"[DART] {ticker} {year}년 {report_type} 실패: {e}")
            continue
    
    return {}


def _parse_finstate(df) -> dict:
    """재무제표 DataFrame에서 핵심 계정 추출."""
    result = {}
    for _, row in df.iterrows():
        acct = str(row.get("account_nm", "")).strip()
        if acct not in ACCOUNT_MAP:
            continue
        key = ACCOUNT_MAP[acct]
        if key in result:
            continue
        val_str = str(row.get("thstrm_amount", "0")).replace(",", "").strip()
        try:
            result[key] = round(float(val_str) / 1e8, 1)  # 원 → 억원
        except ValueError:
            result[key] = 0.0
    return result


def _calc_ratios(data: dict) -> dict:
    """핵심 재무비율 계산."""
    rev         = data.get("revenue", 0)
    op          = data.get("operating_income", 0)
    net         = data.get("net_income", 0)
    assets      = data.get("total_assets", 0)
    liab        = data.get("total_liabilities", 0)
    equity      = data.get("total_equity", 0)
    cur_assets  = data.get("current_assets", 0)
    cur_liab    = data.get("current_liabilities", 0)
    interest    = data.get("interest_expense", 0)

    data["roe"]        = round(net / equity * 100, 2) if equity else 0.0
    data["roa"]        = round(net / assets * 100, 2) if assets else 0.0
    data["debt_ratio"] = round(liab / equity * 100, 2) if equity else 0.0
    data["op_margin"]  = round(op / rev * 100, 2) if rev else 0.0
    data["net_margin"] = round(net / rev * 100, 2) if rev else 0.0
    
    # 안정성 지표
    data["current_ratio"] = round(cur_assets / cur_liab * 100, 1) if cur_liab else 0.0
    data["interest_coverage"] = round(op / interest, 1) if interest > 0 else 999.9
    
    return data


def _calc_growth_rates(data: dict) -> dict:
    """전년 대비 성장률 계산."""
    current = data
    previous = data.get("previous", {})
    
    if not previous:
        return data
    
    # 매출 성장률
    cur_rev = current.get("revenue", 0)
    prev_rev = previous.get("revenue", 0)
    if prev_rev > 0:
        data["revenue_growth"] = round((cur_rev - prev_rev) / prev_rev * 100, 1)
    else:
        data["revenue_growth"] = 0.0
    
    # 영업이익 성장률
    cur_op = current.get("operating_income", 0)
    prev_op = previous.get("operating_income", 0)
    if prev_op > 0:
        data["operating_growth"] = round((cur_op - prev_op) / prev_op * 100, 1)
    else:
        data["operating_growth"] = 0.0 if cur_op >= 0 else -999.0
    
    # 순이익 성장률
    cur_net = current.get("net_income", 0)
    prev_net = previous.get("net_income", 0)
    if prev_net > 0:
        data["net_growth"] = round((cur_net - prev_net) / prev_net * 100, 1)
    else:
        data["net_growth"] = 0.0 if cur_net >= 0 else -999.0
    
    return data


def format_dart_html(data: dict) -> str:
    """재무데이터를 HTML 드롭다운으로 포맷. make_table에서 호출."""
    if not data:
        return "<span style='color:#555'>-</span>"

    year  = data.get("year", "")
    rtype = data.get("report_type", "")

    def _fmt(v):
        if not v: return "-"
        if abs(v) >= 10000:
            return f"{v/10000:,.1f}조"
        return f"{v:,.0f}억"

    def _pct(v):
        if not v: return "-"
        color = "#e53935" if v > 0 else "#1e88e5"
        return f'<span style="color:{color}">{v:+.1f}%</span>'

    rows = [
        ("매출액",      _fmt(data.get("revenue", 0))),
        ("영업이익",    _fmt(data.get("operating_income", 0))),
        ("순이익",      _fmt(data.get("net_income", 0))),
        ("───────",    "───────"),
        ("매출성장률",  _pct(data.get("revenue_growth", 0))),
        ("영업성장률",  _pct(data.get("operating_growth", 0))),
        ("───────",    "───────"),
        ("영업이익률",  _pct(data.get("op_margin", 0))),
        ("ROE",         _pct(data.get("roe", 0))),
        ("ROA",         _pct(data.get("roa", 0))),
        ("───────",    "───────"),
        ("부채비율",    f"{data.get('debt_ratio', 0):.1f}%"),
        ("유동비율",    f"{data.get('current_ratio', 0):.1f}%"),
        ("이자보상배율", f"{data.get('interest_coverage', 0):.1f}"),
    ]

    inner = "".join(
        f"<tr>"
        f"<td style='color:#8b949e;font-size:0.8em;padding:2px 8px'>{k}</td>"
        f"<td style='text-align:right;font-size:0.85em;padding:2px 8px'>{v}</td>"
        f"</tr>"
        for k, v in rows
    )

    return (
        f"<details>"
        f"<summary style='cursor:pointer;color:#58a6ff;font-size:0.8em'>"
        f"{year} {rtype}</summary>"
        f"<table style='width:100%;margin-top:4px'>{inner}</table>"
        f"</details>"
    )
