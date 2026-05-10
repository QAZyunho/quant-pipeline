# 📦 퀀트 파이프라인 설치 및 시작 가이드

## 1. 설치

```bash
pip install pykrx yfinance pandas numpy pyarrow
```

## 2. 디렉토리 구조

```
quant_pipeline/
├── data_pipeline.py   # 한미 데이터 수집
├── indicators.py      # 기술적 지표 계산
├── data/              # 자동 생성됨
│   ├── kr/            # 한국 데이터
│   ├── us/            # 미국 데이터
│   └── cache/         # 캐시 (parquet)
└── pipeline.log       # 실행 로그
```

## 3. 빠른 시작

### 단일 종목 조회
```bash
# 한국 (6자리 숫자 = 자동 감지)
python data_pipeline.py --ticker 005930   # 삼성전자
python data_pipeline.py --ticker 000660   # SK하이닉스

# 미국
python data_pipeline.py --ticker AAPL
python data_pipeline.py --ticker NVDA
python data_pipeline.py --ticker SPY      # ETF
```

### 전체 관심 종목 갱신
```bash
python data_pipeline.py --daily
```

### 상관관계 행렬
```bash
python data_pipeline.py --corr
```

## 4. 코드에서 사용

```python
from data_pipeline import DataPipeline
from indicators import add_indicators, get_signal_summary

# 파이프라인 초기화 (관심 종목 커스텀 가능)
pipeline = DataPipeline(
    kr_tickers=["005930", "000660", "035420"],
    us_tickers=["AAPL", "NVDA", "SPY"],
)

# OHLCV 수집
df_samsung = pipeline.get_ohlcv("005930")
df_apple   = pipeline.get_ohlcv("AAPL")

# 지표 추가
df_samsung = add_indicators(df_samsung)

# 현재 신호 확인
summary = get_signal_summary(df_samsung)
print(summary["overall"])    # "buy" / "sell" / "neutral"
print(summary["buy_count"])  # 매수 신호 개수 (7개 중)
```

## 5. 일별 자동 실행 (cron)

```bash
# crontab -e 에 추가
# 평일 오전 9시 (장 시작 전) 갱신
0 9 * * 1-5 cd /your/path && python data_pipeline.py --daily
```

## 6. 다음 단계

| 모듈 | 설명 |
|------|------|
| `hmm_model.py` | HMM 시장 체제 감지 (다음 단계) |
| `factor_model.py` | 멀티팩터 스코어링 |
| `paper_trading.py` | LLM 페이퍼 트레이딩 봇 |
| `dashboard.py` | Streamlit 대시보드 |

## 7. 주요 컬럼 설명

### OHLCV 기본
| 컬럼 | 설명 |
|------|------|
| `returns` | 일간 수익률 |
| `log_returns` | 로그 수익률 |
| `range` | (고가-저가)/종가 (변동폭) |
| `vol_change` | 거래량 변화율 |

### 지표 (`add_indicators` 후)
| 컬럼 | 설명 |
|------|------|
| `rsi` | RSI (14일) |
| `macd_hist` | MACD 히스토그램 |
| `adx` | 추세 강도 |
| `bb_pct` | 볼린저 밴드 위치 (0~1) |
| `mom_score` | 1M/3M/6M 모멘텀 복합 |
| `returns_z` | **HMM 피처 1** - 정규화 수익률 |
| `range_z` | **HMM 피처 2** - 정규화 변동폭 |
| `vol_change_z` | **HMM 피처 3** - 정규화 거래량 변화 |
