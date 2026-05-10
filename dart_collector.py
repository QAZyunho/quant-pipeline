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

사용법:
    from dart_collector import get_dart_financials
    data = get_dart_financials("005930")  # 삼성전자
"""

import os
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# 보고서 코드
REPORT_CODES = {
    "annual":  "11011",  # 사업보고서 (연간)
    "q1":      "11013",  # 1분기보고서
    "q2":      "11012",  # 반기보고서
    "q3":      "11014",  # 3분기보고서
}

# 재무제표 계정명 매핑 (DART 계정명 → 우리 키)
ACCOUNT_MAP = {
    # 손익계산서
    "매출액":               "revenue",
    "수익(매출액)":          "revenue",
    "영업이익":              "operating_income",
    "영업이익(손실)":        "operating_income",
    "당기순이익":            "net_income",
    "당기순이익(손실)":      "net_income",
    "법인세비용차감전순이익": "pretax_income",
    # 재무상태표
    "자산총계":              "total_assets",
    "부채총계":              "total_liabilities",
    "자본총계":              "total_equity",
    "자본금":                "capital",
}


def get_dart_financials(
    ticker: str,
    api_key: Optional[str] = None,
    year: Optional[int] = None,
) -> dict:
    """
    OpenDartReader로 재무제표 핵심 지표 수집.

    Args:
        ticker:  종목코드 (예: "005930")
        api_key: DART API 키 (없으면 환경변수 DART_API_KEY 사용)
        year:    수집 연도 (없으면 최근 연도 자동 선택)

    Returns:
        {
            "revenue":          매출액 (억원),
            "operating_income": 영업이익 (억원),
            "net_income":       당기순이익 (억원),
            "total_assets":     자산총계 (억원),
            "total_liabilities":부채총계 (억원),
            "total_equity":     자본총계 (억원),
            "roe":              ROE (%),
            "roa":              ROA (%),
            "debt_ratio":       부채비율 (%),
            "op_margin":        영업이익률 (%),
            "net_margin":       순이익률 (%),
            "year":             기준연도,
            "report_type":      보고서 종류,
        }
    """
    key = api_key or os.environ.get("DART_API_KEY", "")
    if not key:
        log.warning(f"[DART] API 키 없음 → DART_API_KEY 환경변수 설정 필요")
        return {}

    try:
        import OpenDartReader
        dart = OpenDartReader(key)
    except Exception as e:
        log.error(f"[DART] OpenDartReader 초기화 실패: {e}")
        return {}

    # 연도 결정 (1~3월은 전전년도, 4월 이후는 전년도 사업보고서가 최신)
    if year is None:
        now = datetime.today()
        year = now.year - 1 if now.month >= 4 else now.year - 2

    # 사업보고서(연간) → 없으면 반기 시도
    for report_type, code in [("연간", "11011"), ("반기", "11012"), ("1분기", "11013")]:
        try:
            df = dart.finstate_all(ticker, year, reprt_code=code, fs_div="CFS")
            if df is not None and not df.empty:
                log.info(f"[DART] {ticker} {year}년 {report_type} 재무제표 수집 완료")
                result = _parse_finstate(df)
                result["year"] = year
                result["report_type"] = report_type
                return _calc_ratios(result)
        except Exception as e:
            log.debug(f"[DART] {ticker} {report_type} 실패: {e}")
            continue

    # 연결재무제표 없으면 별도재무제표 시도
    for report_type, code in [("연간(별도)", "11011"), ("반기(별도)", "11012")]:
        try:
            df = dart.finstate_all(ticker, year, reprt_code=code, fs_div="OFS")
            if df is not None and not df.empty:
                log.info(f"[DART] {ticker} {year}년 {report_type} 재무제표 수집 완료")
                result = _parse_finstate(df)
                result["year"] = year
                result["report_type"] = report_type
                return _calc_ratios(result)
        except Exception as e:
            log.debug(f"[DART] {ticker} {report_type}(별도) 실패: {e}")
            continue

    log.warning(f"[DART] {ticker} 재무제표 수집 실패")
    return {}


def _parse_finstate(df) -> dict:
    """재무제표 DataFrame에서 핵심 계정 추출."""
    result = {}

    for _, row in df.iterrows():
        acct = str(row.get("account_nm", "")).strip()
        if acct not in ACCOUNT_MAP:
            continue

        key = ACCOUNT_MAP[acct]
        if key in result:  # 이미 있으면 스킵 (중복 계정 처리)
            continue

        # 당기 금액 (thstrm_amount)
        val_str = str(row.get("thstrm_amount", "0")).replace(",", "").strip()
        try:
            val = float(val_str) / 1e8  # 원 → 억원
            result[key] = round(val, 1)
        except ValueError:
            result[key] = 0.0

    return result


def _calc_ratios(data: dict) -> dict:
    """핵심 재무비율 계산."""
    rev   = data.get("revenue", 0)
    op    = data.get("operating_income", 0)
    net   = data.get("net_income", 0)
    assets = data.get("total_assets", 0)
    liab  = data.get("total_liabilities", 0)
    equity = data.get("total_equity", 0)

    # ROE = 당기순이익 / 자본총계 * 100
    data["roe"] = round(net / equity * 100, 2) if equity else 0.0

    # ROA = 당기순이익 / 자산총계 * 100
    data["roa"] = round(net / assets * 100, 2) if assets else 0.0

    # 부채비율 = 부채총계 / 자본총계 * 100
    data["debt_ratio"] = round(liab / equity * 100, 2) if equity else 0.0

    # 영업이익률
    data["op_margin"] = round(op / rev * 100, 2) if rev else 0.0

    # 순이익률
    data["net_margin"] = round(net / rev * 100, 2) if rev else 0.0

    return data


def format_dart_html(data: dict) -> str:
    """재무데이터를 HTML 미니 테이블로 포맷."""
    if not data:
        return "<span style='color:#555'>-</span>"

    year = data.get("year", "")
    rtype = data.get("report_type", "")

    def _fmt(v, unit="억"):
        if v == 0: return "-"
        if abs(v) >= 10000:
            return f"{v/10000:,.1f}조"
        return f"{v:,.0f}{unit}"

    def _pct(v):
        if v == 0: return "-"
        color = "#e53935" if v > 0 else "#1e88e5"
        return f'<span style="color:{color}">{v:+.1f}%</span>'

    rows = [
        ("매출액",    _fmt(data.get("revenue", 0))),
        ("영업이익",  _fmt(data.get("operating_income", 0))),
        ("순이익",    _fmt(data.get("net_income", 0))),
        ("영업이익률", _pct(data.get("op_margin", 0))),
        ("ROE",       _pct(data.get("roe", 0))),
        ("ROA",       _pct(data.get("roa", 0))),
        ("부채비율",  f"{data.get('debt_ratio', 0):.1f}%"),
    ]

    inner = "".join(
        f"<tr><td style='color:#8b949e;font-size:0.8em'>{k}</td>"
        f"<td style='text-align:right;font-size:0.85em'>{v}</td></tr>"
        for k, v in rows
    )

    return f"""
    <details>
      <summary style='cursor:pointer;color:#58a6ff;font-size:0.8em'>
        재무제표 {year} {rtype}
      </summary>
      <table style='width:100%;margin-top:4px'>{inner}</table>
    </details>"""
