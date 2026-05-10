"""
키움 REST API 클라이언트
========================
문서 기반으로 구현한 키움증권 REST API 연동 모듈.

지원 API:
  - au10001: OAuth 접근토큰 발급
  - ka10001: 주식기본정보요청 (현재가, PER, PBR 등)
  - ka10081: 주식일봉차트조회요청 (OHLCV)
  - ka10095: 관심종목정보요청 (여러 종목 현재가 일괄)

사용법:
    from kiwoom_client import KiwoomClient

    client = KiwoomClient(
        appkey=os.environ["KIWOOM_APPKEY"],
        secretkey=os.environ["KIWOOM_SECRETKEY"],
    )
    df = client.get_ohlcv("005930", years=2)

GitHub Actions에서는 Secrets에 저장:
    KIWOOM_APPKEY, KIWOOM_SECRETKEY
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests

log = logging.getLogger(__name__)

# ── 도메인 ────────────────────────────────────────────────────
PROD_DOMAIN = "https://api.kiwoom.com"
MOCK_DOMAIN = "https://mockapi.kiwoom.com"

# ── 토큰 캐시 경로 ────────────────────────────────────────────
TOKEN_CACHE = Path(".kiwoom_token.json")


class KiwoomAuthError(Exception):
    pass


class KiwoomAPIError(Exception):
    pass


# ════════════════════════════════════════════════════════════
# 키움 REST API 클라이언트
# ════════════════════════════════════════════════════════════
class KiwoomClient:
    """
    키움 REST API 클라이언트.

    Args:
        appkey:    키움 OpenAPI 앱키 (환경변수 KIWOOM_APPKEY 권장)
        secretkey: 키움 OpenAPI 시크릿키 (환경변수 KIWOOM_SECRETKEY 권장)
        mock:      True면 모의투자 도메인 사용 (기본 False)
    """

    def __init__(
        self,
        appkey: Optional[str] = None,
        secretkey: Optional[str] = None,
        mock: bool = False,
    ):
        self.appkey    = appkey    or os.environ.get("KIWOOM_APPKEY", "")
        self.secretkey = secretkey or os.environ.get("KIWOOM_SECRETKEY", "")
        self.base_url  = MOCK_DOMAIN if mock else PROD_DOMAIN
        self.mock      = mock
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

        if not self.appkey or not self.secretkey:
            raise KiwoomAuthError(
                "KIWOOM_APPKEY, KIWOOM_SECRETKEY 환경변수를 설정하거나 "
                "KiwoomClient(appkey=..., secretkey=...) 로 직접 전달하세요."
            )

        log.info(f"KiwoomClient 초기화 ({'모의투자' if mock else '실전'})")
        self._load_token_cache()

    # ════════════════════════════════════════════════════════
    # 인증 (au10001)
    # ════════════════════════════════════════════════════════
    def _load_token_cache(self):
        """로컬 캐시에서 토큰 불러오기."""
        if not TOKEN_CACHE.exists():
            return
        try:
            data = json.loads(TOKEN_CACHE.read_text())
            expires = datetime.strptime(data["expires_dt"], "%Y%m%d%H%M%S")
            if expires > datetime.now() + timedelta(minutes=10):
                self._token = data["token"]
                self._token_expires = expires
                log.debug("토큰 캐시 로드 완료")
        except Exception:
            pass

    def _save_token_cache(self, token: str, expires_dt: str):
        TOKEN_CACHE.write_text(json.dumps({
            "token": token,
            "expires_dt": expires_dt,
        }))

    def _get_token(self) -> str:
        """유효한 토큰 반환. 만료됐으면 재발급."""
        if self._token and self._token_expires:
            if self._token_expires > datetime.now() + timedelta(minutes=10):
                return self._token

        log.info("접근토큰 발급 중... (au10001)")
        resp = requests.post(
            f"{self.base_url}/oauth2/token",
            json={
                "grant_type": "client_credentials",
                "appkey": self.appkey,
                "secretkey": self.secretkey,
            },
            headers={"Content-Type": "application/json;charset=UTF-8"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("return_code", -1) != 0:
            raise KiwoomAuthError(f"토큰 발급 실패: {data.get('return_msg')}")

        self._token = data["token"]
        self._token_expires = datetime.strptime(data["expires_dt"], "%Y%m%d%H%M%S")
        self._save_token_cache(self._token, data["expires_dt"])
        log.info(f"토큰 발급 완료. 만료: {self._token_expires}")
        return self._token

    # ════════════════════════════════════════════════════════
    # 공통 요청 헬퍼
    # ════════════════════════════════════════════════════════
    def _post(
        self,
        url: str,
        api_id: str,
        body: dict,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """키움 REST API 공통 POST 요청."""
        token = self._get_token()
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": api_id,
        }
        if cont_yn == "Y":
            headers["cont-yn"] = cont_yn
            headers["next-key"] = next_key

        resp = requests.post(
            f"{self.base_url}{url}",
            json=body,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # 연속조회 여부를 응답에 포함
        data["_cont_yn"]  = resp.headers.get("cont-yn", "N")
        data["_next_key"] = resp.headers.get("next-key", "")
        return data

    # ════════════════════════════════════════════════════════
    # ka10081: 주식일봉차트조회
    # ════════════════════════════════════════════════════════
    def get_ohlcv(
        self,
        ticker: str,
        years: int = 2,
        adj_price: bool = True,
    ) -> pd.DataFrame:
        """
        일봉 OHLCV 데이터 수집 (연속조회로 전체 기간 가져옴).

        Args:
            ticker:    종목코드 (예: "005930")
            years:     수집 기간 (연)
            adj_price: 수정주가 여부 (기본 True)

        Returns:
            DataFrame: date 인덱스, open/high/low/close/volume 컬럼
        """
        base_dt = datetime.today().strftime("%Y%m%d")
        cutoff  = datetime.today() - timedelta(days=365 * years)
        upd_tp  = "1" if adj_price else "0"

        all_rows = []
        cont_yn  = "N"
        next_key = ""

        log.info(f"[KR] {ticker} 일봉 수집 중... ({years}년치)")

        while True:
            data = self._post(
                url="/api/dostk/chart",
                api_id="ka10081",
                body={"stk_cd": ticker, "base_dt": base_dt, "upd_stkpc_tp": upd_tp},
                cont_yn=cont_yn,
                next_key=next_key,
            )

            rows = data.get("stk_dt_pole_chart_qry", [])
            if not rows:
                break

            for r in rows:
                dt = r.get("dt", "")
                if len(dt) < 8:
                    continue
                row_date = datetime.strptime(dt[:8], "%Y%m%d")
                if row_date < cutoff:
                    # 목표 기간 이전 데이터 → 수집 종료
                    all_rows.append(r)
                    cont_yn = "N"
                    break
                all_rows.append(r)

            cont_yn  = data.get("_cont_yn", "N")
            next_key = data.get("_next_key", "")

            if cont_yn != "Y":
                break

            time.sleep(0.3)  # API 과부하 방지

        if not all_rows:
            log.warning(f"[KR] {ticker} 데이터 없음")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # 컬럼 정리
        col_map = {
            "dt":         "date",
            "open_pric":  "open",
            "high_pric":  "high",
            "low_pric":   "low",
            "cur_prc":    "close",
            "trde_qty":   "volume",
        }
        df = df.rename(columns=col_map)
        df = df[list(col_map.values())].copy()

        # 타입 변환 (키움은 문자열로 반환, 부호 제거)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace("+", "")
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

        df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d")
        df = df.set_index("date").sort_index()
        df = df[df.index >= cutoff]

        # 파생 컬럼
        df["returns"]     = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["range"]       = (df["high"] - df["low"]) / df["close"]
        df["vol_change"]  = df["volume"].pct_change()
        df["ticker"]      = ticker
        df["market"]      = "KR"

        log.info(f"[KR] {ticker} 수집 완료: {len(df)}행 ({df.index[0].date()} ~ {df.index[-1].date()})")
        return df.dropna()

    # ════════════════════════════════════════════════════════
    # ka10001: 주식기본정보 (현재가, PER, PBR 등)
    # ════════════════════════════════════════════════════════
    def get_stock_info(self, ticker: str) -> dict:
        """
        종목 기본정보 조회.
        현재가, PER, PBR, EPS, 시총, 52주 고저 등.
        """
        log.info(f"[KR] {ticker} 기본정보 조회")
        data = self._post(
            url="/api/dostk/stkinfo",
            api_id="ka10001",
            body={"stk_cd": ticker},
        )
        info = {
            "ticker":       ticker,
            "name":         data.get("stk_nm", ""),
            "close":        _to_num(data.get("cur_prc")),
            "market_cap":   _to_num(data.get("mkt_cap")),
            "per":          _to_num(data.get("per")),
            "pbr":          _to_num(data.get("pbr")),
            "eps":          _to_num(data.get("eps")),
            "week52_high":  _to_num(data.get("high_pric_52")),
            "week52_low":   _to_num(data.get("low_pric_52")),
            "shares":       _to_num(data.get("lstg_stk_cnt")),
        }
        return info

    # ════════════════════════════════════════════════════════
    # ka10095: 관심종목정보 (여러 종목 현재가 일괄)
    # ════════════════════════════════════════════════════════
    def get_watchlist_quotes(self, tickers: list[str]) -> pd.DataFrame:
        """
        여러 종목의 현재가/등락률/거래량을 한번에 조회.
        watchlist.json 종목들을 빠르게 스냅샷할 때 사용.

        Args:
            tickers: 종목코드 리스트 (예: ["005930", "000660"])

        Returns:
            DataFrame: ticker, name, close, change_pct, volume 컬럼
        """
        # | 로 구분해서 전달 (문서 스펙)
        stk_cd = "|".join(tickers)
        log.info(f"[KR] 관심종목 일괄 조회: {len(tickers)}개")

        data = self._post(
            url="/api/dostk/stkinfo",
            api_id="ka10095",
            body={"stk_cd": stk_cd},
        )

        rows = data.get("atn_stk_infr", [])
        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            records.append({
                "ticker":     r.get("stk_cd", ""),
                "name":       r.get("stk_nm", ""),
                "close":      _to_num(r.get("cur_prc")),
                "change_pct": _to_num(r.get("flu_rt")),
                "volume":     _to_num(r.get("trde_qty")),
            })
        return pd.DataFrame(records)

    # ════════════════════════════════════════════════════════
    # 배치 수집 (data_pipeline 호환)
    # ════════════════════════════════════════════════════════
    def get_batch(
        self,
        tickers: list[str],
        years: int = 2,
        delay: float = 0.5,
    ) -> dict:
        """
        여러 종목 OHLCV + 기본정보 일괄 수집.
        DataPipeline의 kr_collector.get_batch() 대체용.
        """
        result = {}
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            log.info(f"[KR] ({i}/{total}) {ticker} 수집 중...")
            try:
                result[ticker] = {
                    "ohlcv": self.get_ohlcv(ticker, years),
                    "info":  self.get_stock_info(ticker),
                }
            except Exception as e:
                log.error(f"[KR] {ticker} 실패: {e}")
                result[ticker] = {"ohlcv": pd.DataFrame(), "info": {}}
            time.sleep(delay)
        return result


# ── 유틸 ──────────────────────────────────────────────────────
def _to_num(val) -> float:
    """키움 응답값(문자열) → float 변환."""
    if val is None:
        return 0.0
    return float(str(val).replace(",", "").replace("+", "").replace("%", "") or 0)


# ════════════════════════════════════════════════════════════
# 간단한 동작 테스트
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 환경변수 확인
    if not os.environ.get("KIWOOM_APPKEY"):
        print("❌ 환경변수를 먼저 설정하세요:")
        print("   export KIWOOM_APPKEY=your_appkey")
        print("   export KIWOOM_SECRETKEY=your_secretkey")
        sys.exit(1)

    client = KiwoomClient()

    # 삼성전자 일봉
    df = client.get_ohlcv("005930", years=1)
    print(f"\n삼성전자 일봉 ({len(df)}행):")
    print(df.tail(3))

    # 관심종목 현재가
    quotes = client.get_watchlist_quotes(["005930", "000660", "035420"])
    print(f"\n관심종목 현재가:")
    print(quotes.to_string(index=False))
