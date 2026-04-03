"""
creditcard.csv 원본 컬럼·값 형식을 유지한 채 행만 줄여 .xlsx를 만듭니다.
fds_pipeline.CSV_DEFAULT와 동일한 파일을 입력으로 씁니다.
"""
from pathlib import Path

import pandas as pd

# 행 수 (파일이 더 짧으면 전부 사용)
N_ROWS = 100

# "head" : CSV 파일의 위에서부터 N_ROWS행 (순서·내용 그대로)
# "random" : 재현 가능한 무작위 표본(random_state=42), 분포가 원본에 가깝게
SAMPLE_MODE = "random"

CSV_IN = Path(__file__).resolve().parent / "dataset" / "creditcard.csv"
XLSX_OUT = Path(__file__).resolve().parent / "dataset" / "creditcard_sample_100.xlsx"


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
