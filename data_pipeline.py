"""
한미 주식 데이터 수집 파이프라인
====================================
한국: pykrx (KRX 공식 데이터)
미국: yfinance (Yahoo Finance)

사용법:
    python data_pipeline.py
    
또는 모듈로 임포트:
    from data_pipeline import KRDataCollector, USDataCollector, DataPipeline
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

# ── 로깅 설정 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── 데이터 저장 경로 ─────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "kr").mkdir(exist_ok=True)
(DATA_DIR / "us").mkdir(exist_ok=True)
(DATA_DIR / "cache").mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# 1. 한국 시장 수집기
# ════════════════════════════════════════════════════════════
class KRDataCollector:
    """
    pykrx 기반 한국 주식 데이터 수집기.
    - OHLCV (일봉)
    - 시가총액 / 외국인 보유율
    - 재무 지표 (PER, PBR, EPS, BPS, DIV)
    """

    def __init__(self, cache: bool = True):
        try:
            import pykrx.stock as krx
            self.krx = krx
        except ImportError:
            raise ImportError("pip install pykrx 를 먼저 실행하세요.")
        self.cache = cache
        log.info("KRDataCollector 초기화 완료")

    # ── 날짜 헬퍼 ────────────────────────────────────────────
    @staticmethod
    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%d")

    @staticmethod
    def _date_range(years: int = 2) -> tuple[str, str]:
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        return KRDataCollector._fmt(start), KRDataCollector._fmt(end)

    # ── 캐시 ─────────────────────────────────────────────────
    def _cache_path(self, ticker: str, suffix: str) -> Path:
        return DATA_DIR / "cache" / f"kr_{ticker}_{suffix}.parquet"

    def _is_fresh(self, path: Path, hours: int = 12) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < hours * 3600

    # ── OHLCV ────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        years: int = 2,
        force: bool = False,
    ) -> pd.DataFrame:
        """일봉 OHLCV + 수익률 컬럼 추가."""
        cache_path = self._cache_path(ticker, "ohlcv")
        if self.cache and not force and self._is_fresh(cache_path):
            log.debug(f"[KR] {ticker} OHLCV 캐시 사용")
            return pd.read_parquet(cache_path)

        start, end = self._date_range(years)
        log.info(f"[KR] {ticker} OHLCV 수집 중... ({start} ~ {end})")

        df = self.krx.get_market_ohlcv(start, end, ticker)
        if df.empty:
            log.warning(f"[KR] {ticker} 데이터 없음")
            return df

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.columns = ["open", "high", "low", "close", "volume", "change_pct"]

        # 파생 컬럼
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["range"] = (df["high"] - df["low"]) / df["close"]
        df["vol_change"] = df["volume"].pct_change()
        df["ticker"] = ticker
        df["market"] = "KR"

        df.dropna(inplace=True)

        if self.cache:
            df.to_parquet(cache_path)
        return df

    # ── 펀더멘털 ─────────────────────────────────────────────
    def get_fundamental(
        self,
        ticker: str,
        years: int = 2,
        force: bool = False,
    ) -> pd.DataFrame:
        """PER, PBR, EPS, BPS, DIV, 시가총액."""
        cache_path = self._cache_path(ticker, "fundamental")
        if self.cache and not force and self._is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        start, end = self._date_range(years)
        log.info(f"[KR] {ticker} 펀더멘털 수집 중...")

        try:
            df_val = self.krx.get_market_fundamental(start, end, ticker)
            df_cap = self.krx.get_market_cap(start, end, ticker)
        except Exception as e:
            log.error(f"[KR] {ticker} 펀더멘털 수집 실패: {e}")
            return pd.DataFrame()

        if df_val.empty:
            return df_val

        df_val.index = pd.to_datetime(df_val.index)
        df_cap.index = pd.to_datetime(df_cap.index)

        df = df_val.join(df_cap[["시가총액"]], how="left")
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"시가총액": "market_cap"}, inplace=True)
        df.index.name = "date"
        df["ticker"] = ticker

        if self.cache:
            df.to_parquet(cache_path)
        return df

    # ── 배치 수집 ────────────────────────────────────────────
    def get_batch(
        self,
        tickers: list[str],
        years: int = 2,
        delay: float = 0.5,
    ) -> dict[str, dict]:
        """
        여러 종목 일괄 수집.
        반환: {ticker: {"ohlcv": df, "fundamental": df}}
        """
        result = {}
        total = len(tickers)
        for i, t in enumerate(tickers, 1):
            log.info(f"[KR] 수집 중 ({i}/{total}): {t}")
            result[t] = {
                "ohlcv": self.get_ohlcv(t, years),
                "fundamental": self.get_fundamental(t, years),
            }
            time.sleep(delay)  # KRX 과부하 방지
        return result

    # ── 종목 목록 ────────────────────────────────────────────
    def get_kospi_tickers(self) -> list[str]:
        """코스피 전체 종목 코드 반환."""
        today = self._fmt(datetime.today())
        return self.krx.get_market_ticker_list(today, market="KOSPI")

    def get_kosdaq_tickers(self) -> list[str]:
        """코스닥 전체 종목 코드 반환."""
        today = self._fmt(datetime.today())
        return self.krx.get_market_ticker_list(today, market="KOSDAQ")

    def get_ticker_name(self, ticker: str) -> str:
        """종목 코드 → 종목명."""
        return self.krx.get_market_ticker_name(ticker)


# ════════════════════════════════════════════════════════════
# 2. 미국 시장 수집기
# ════════════════════════════════════════════════════════════
class USDataCollector:
    """
    yfinance 기반 미국 주식/ETF 데이터 수집기.
    - OHLCV (일봉)
    - 기업 정보 및 재무 요약
    - 옵션 내재 변동성 (선택)
    """

    def __init__(self, cache: bool = True):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("pip install yfinance 를 먼저 실행하세요.")
        self.cache = cache
        log.info("USDataCollector 초기화 완료")

    def _cache_path(self, ticker: str, suffix: str) -> Path:
        return DATA_DIR / "cache" / f"us_{ticker}_{suffix}.parquet"

    def _is_fresh(self, path: Path, hours: int = 12) -> bool:
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) < hours * 3600

    # ── OHLCV ────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        years: int = 2,
        force: bool = False,
    ) -> pd.DataFrame:
        """일봉 OHLCV + 파생 지표."""
        cache_path = self._cache_path(ticker, "ohlcv")
        if self.cache and not force and self._is_fresh(cache_path):
            log.debug(f"[US] {ticker} OHLCV 캐시 사용")
            return pd.read_parquet(cache_path)

        end = datetime.today()
        start = end - timedelta(days=365 * years)
        log.info(f"[US] {ticker} OHLCV 수집 중...")

        t = self.yf.Ticker(ticker)
        df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if df.empty:
            log.warning(f"[US] {ticker} 데이터 없음")
            return df

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.rename(columns={"stock_splits": "splits", "capital_gains": "cap_gains"}, errors="ignore", inplace=True)

        # 파생 컬럼
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["range"] = (df["high"] - df["low"]) / df["close"]
        df["vol_change"] = df["volume"].pct_change()
        df["ticker"] = ticker
        df["market"] = "US"

        df.dropna(inplace=True)

        if self.cache:
            df.to_parquet(cache_path)
        return df

    # ── 기업 정보 ────────────────────────────────────────────
    def get_info(self, ticker: str) -> dict:
        """yfinance .info 딕셔너리 (PER, 시총, 섹터 등)."""
        cache_path = DATA_DIR / "cache" / f"us_{ticker}_info.json"
        if self.cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < 86400:  # 1일 캐시
                with open(cache_path) as f:
                    return json.load(f)

        log.info(f"[US] {ticker} info 수집 중...")
        info = self.yf.Ticker(ticker).info

        key_fields = [
            "shortName", "sector", "industry", "marketCap",
            "trailingPE", "forwardPE", "priceToBook",
            "returnOnEquity", "returnOnAssets", "debtToEquity",
            "revenueGrowth", "earningsGrowth", "dividendYield",
            "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        filtered = {k: info.get(k) for k in key_fields}

        if self.cache:
            with open(cache_path, "w") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
        return filtered

    # ── 배치 수집 ────────────────────────────────────────────
    def get_batch(
        self,
        tickers: list[str],
        years: int = 2,
        delay: float = 0.3,
    ) -> dict[str, dict]:
        result = {}
        total = len(tickers)
        for i, t in enumerate(tickers, 1):
            log.info(f"[US] 수집 중 ({i}/{total}): {t}")
            try:
                result[t] = {
                    "ohlcv": self.get_ohlcv(t, years),
                    "info": self.get_info(t),
                }
            except Exception as e:
                log.error(f"[US] {t} 수집 실패: {e}")
                result[t] = {"ohlcv": pd.DataFrame(), "info": {}}
            time.sleep(delay)
        return result


# ════════════════════════════════════════════════════════════
# 3. 통합 파이프라인
# ════════════════════════════════════════════════════════════
class DataPipeline:
    """
    한미 데이터를 통합 관리하는 메인 파이프라인.
    
    사용 예시:
        pipeline = DataPipeline()
        pipeline.run_daily()                 # 매일 전체 갱신
        df = pipeline.get_ohlcv("005930")    # 한국
        df = pipeline.get_ohlcv("AAPL")      # 미국 (자동 감지)
    """

    # 기본 관심 종목
    DEFAULT_KR = [
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "035420",  # NAVER
        "005380",  # 현대차
        "051910",  # LG화학
        "035720",  # 카카오
        "207940",  # 삼성바이오로직스
        "006400",  # 삼성SDI
    ]

    DEFAULT_US = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "BRK-B",
        # ETF
        "SPY", "QQQ", "IWM", "TLT", "GLD",
    ]

    def __init__(
        self,
        kr_tickers: Optional[list[str]] = None,
        us_tickers: Optional[list[str]] = None,
    ):
        self.kr_collector = KRDataCollector()
        self.us_collector = USDataCollector()
        self.kr_tickers = kr_tickers or self.DEFAULT_KR
        self.us_tickers = us_tickers or self.DEFAULT_US
        self._kr_data: dict = {}
        self._us_data: dict = {}

    # ── 자동 시장 감지 ───────────────────────────────────────
    def _is_kr(self, ticker: str) -> bool:
        """숫자 6자리면 한국 종목으로 판단."""
        return ticker.isdigit() and len(ticker) == 6

    def get_ohlcv(self, ticker: str, years: int = 2) -> pd.DataFrame:
        """한/미 자동 감지 후 OHLCV 반환."""
        if self._is_kr(ticker):
            return self.kr_collector.get_ohlcv(ticker, years)
        return self.us_collector.get_ohlcv(ticker, years)

    # ── 일별 갱신 ────────────────────────────────────────────
    def run_daily(self, years: int = 2) -> None:
        """전체 관심 종목 데이터를 갱신하고 로컬에 저장."""
        log.info("=" * 50)
        log.info("📡 일별 데이터 갱신 시작")
        log.info(f"   한국: {len(self.kr_tickers)}개 | 미국: {len(self.us_tickers)}개")
        log.info("=" * 50)

        self._kr_data = self.kr_collector.get_batch(self.kr_tickers, years)
        self._us_data = self.us_collector.get_batch(self.us_tickers, years)

        self._save_summary()
        log.info("✅ 일별 갱신 완료")

    # ── 요약 저장 ────────────────────────────────────────────
    def _save_summary(self) -> None:
        """수집 결과 요약을 JSON으로 저장."""
        summary = {
            "updated_at": datetime.now().isoformat(),
            "kr": {},
            "us": {},
        }

        for ticker, data in self._kr_data.items():
            df = data.get("ohlcv", pd.DataFrame())
            if not df.empty:
                summary["kr"][ticker] = {
                    "rows": len(df),
                    "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()),
                    "last_close": float(df["close"].iloc[-1]),
                    "last_return": float(df["returns"].iloc[-1]),
                }

        for ticker, data in self._us_data.items():
            df = data.get("ohlcv", pd.DataFrame())
            if not df.empty:
                summary["us"][ticker] = {
                    "rows": len(df),
                    "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()),
                    "last_close": float(df["close"].iloc[-1]),
                    "last_return": float(df["returns"].iloc[-1]),
                }

        path = DATA_DIR / "pipeline_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info(f"📄 요약 저장: {path}")

    # ── 단일 종목 전체 데이터 ────────────────────────────────
    def get_full(self, ticker: str) -> dict:
        """OHLCV + 펀더멘털 통합 반환."""
        if self._is_kr(ticker):
            return {
                "ohlcv": self.kr_collector.get_ohlcv(ticker),
                "fundamental": self.kr_collector.get_fundamental(ticker),
                "market": "KR",
                "name": self.kr_collector.get_ticker_name(ticker),
            }
        return {
            "ohlcv": self.us_collector.get_ohlcv(ticker),
            "info": self.us_collector.get_info(ticker),
            "market": "US",
            "name": ticker,
        }

    # ── 수익률 상관관계 ──────────────────────────────────────
    def get_correlation_matrix(
        self,
        tickers: Optional[list[str]] = None,
        years: int = 1,
    ) -> pd.DataFrame:
        """
        한미 종목 간 수익률 상관관계 행렬.
        HMM 팩터 분석 전 포트폴리오 분산 검토에 활용.
        """
        tickers = tickers or (self.kr_tickers[:4] + self.us_tickers[:4])
        returns_dict = {}

        for t in tickers:
            df = self.get_ohlcv(t, years)
            if not df.empty:
                returns_dict[t] = df["returns"]

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict).corr().round(3)


# ════════════════════════════════════════════════════════════
# 4. 실행
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="한미 주식 데이터 수집 파이프라인")
    parser.add_argument("--ticker", "-t", type=str, help="단일 종목 조회 (예: 005930, AAPL)")
    parser.add_argument("--years", "-y", type=int, default=2, help="수집 기간 (연, 기본 2)")
    parser.add_argument("--daily", action="store_true", help="전체 관심 종목 일별 갱신")
    parser.add_argument("--corr", action="store_true", help="상관관계 행렬 출력")
    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.ticker:
        log.info(f"단일 종목 조회: {args.ticker}")
        data = pipeline.get_full(args.ticker)
        df = data["ohlcv"]
        if not df.empty:
            print(f"\n{'='*40}")
            print(f"📊 {data.get('name', args.ticker)} [{data['market']}]")
            print(f"{'='*40}")
            print(df.tail(5).to_string())
            print(f"\n최근 30일 평균 수익률: {df['returns'].tail(30).mean()*100:.3f}%")
            print(f"최근 30일 변동성:       {df['returns'].tail(30).std()*100:.3f}%")

    elif args.daily:
        pipeline.run_daily(args.years)

    elif args.corr:
        print("\n📈 수익률 상관관계 행렬")
        print(pipeline.get_correlation_matrix())

    else:
        # 기본: 빠른 데모
        print("\n🚀 파이프라인 테스트")
        print("사용 예시:")
        print("  python data_pipeline.py --ticker 005930   # 삼성전자")
        print("  python data_pipeline.py --ticker AAPL     # 애플")
        print("  python data_pipeline.py --daily           # 전체 갱신")
        print("  python data_pipeline.py --corr            # 상관관계")
