# 2차: RF baseline + XGB baseline + XGB tuned (3종).
# Train을 fit|val로 나눈 뒤 SMOTETomek은 fit에만 적용 → 각 모델의 검증 확률로 F2 최대 임계값 선택 → 동일 test에서만 지표.
# (1차 전체 train 리샘플 학습과 학습 표본이 달라 숫자는 1차 CSV와 직접 비교하지 말 것.)
# python scripts/model_comparison_round2.py [--csv path] [--out report_figures/model_round2_val_thr.csv] [--val-fraction 0.15]
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import repro_threads  # noqa: E402

import pandas as pd

from fds_pipeline import CSV_DEFAULT

_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from model_comparison import run_round2_val_threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3모델: 검증에서 F2 최대 임계값 → hold-out test 평가",
    )
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV 저장 (기본: report_figures/model_round2_val_thr.csv)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="원시 train 중 검증 비율 (층화)",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = _ROOT / "report_figures" / "model_round2_val_thr.csv"

    df, meta = run_round2_val_threshold(args.csv, val_fraction=args.val_fraction)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== Round 2: val에서 F2 최대 임계값 → test 지표 ===\n")
    print(f"설정: val_fraction={meta['val_fraction']}, random_state={meta['random_state']}\n")
    print(df.to_string(index=False))
    if "xgb_tuned_params" in meta:
        print("\n=== XGB tuned best params (요약) ===\n")
        print(meta["xgb_tuned_params"])

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n저장: {out.resolve()}")


if __name__ == "__main__":
    main()
