# creditcard.csv에서 N행만 뽑아 xlsx로 저장.
# python scripts/make_creditcard_sample_excel.py
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

# 행 수 (파일이 더 짧으면 전부 사용)
N_ROWS = 100

# "head" : CSV 파일의 위에서부터 N_ROWS행 (순서·내용 그대로)
# "random" : 재현 가능한 무작위 표본(random_state=42), 분포가 원본에 가깝게
SAMPLE_MODE = "random"

CSV_IN = _ROOT / "dataset" / "creditcard.csv"
XLSX_OUT = _ROOT / "dataset" / "creditcard_sample_100.xlsx"


def main() -> None:
    if not CSV_IN.is_file():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {CSV_IN}")

    df = pd.read_csv(CSV_IN)
    n = min(N_ROWS, len(df))

    if SAMPLE_MODE == "head":
        out = df.head(n)
    elif SAMPLE_MODE == "random":
        out = df.sample(n=n, random_state=42).sort_index()
    else:
        raise ValueError(f"SAMPLE_MODE는 'head' 또는 'random'만 허용: {SAMPLE_MODE!r}")

    out.to_excel(XLSX_OUT, index=False, engine="openpyxl")
    print(f"저장 완료: {len(out)}행 → {XLSX_OUT}")


if __name__ == "__main__":
    main()
