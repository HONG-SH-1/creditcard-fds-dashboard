# 제3장 3-1 슬라이드용: 전처리 후 Time_sin · Time_cos 분포 (변환 후)
# fds_pipeline.preprocess_creditcard_dataframe 과 동일 변환.
# 출력: report_figures/eda/preprocess_time_sin_cos_kde.png
# 실행: 프로젝트 루트에서  python scripts/report_plots/plot_preprocess_time_sin_cos.py
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from fds_pipeline import CSV_DEFAULT, preprocess_creditcard_dataframe
from report.report_figures_common import REPORT_FIGURES_EDA, setup_korean_matplotlib_font

OUT_DIR = _PROJECT_ROOT / REPORT_FIGURES_EDA
OUT_PATH = OUT_DIR / "preprocess_time_sin_cos_kde.png"


def main() -> None:
    setup_korean_matplotlib_font()

    csv_path = Path(CSV_DEFAULT)
    if not csv_path.is_file():
        raise FileNotFoundError(f"데이터가 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    X, _, _ = preprocess_creditcard_dataframe(df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, title in [
        (axes[0], "Time_sin", "Time_sin 분포 (전처리 후)"),
        (axes[1], "Time_cos", "Time_cos 분포 (전처리 후)"),
    ]:
        s = X[col].astype(float).values
        ax.hist(s, bins=80, density=True, color="#1f77b4", alpha=0.55, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(col)
        ax.set_ylabel("밀도")
        ax.set_title(title)

    fig.suptitle(
        "Time → sin/cos 주기 인코딩 후 특성 분포 (86400초 주기, 원 Time 컬럼 삭제 전 단계와 동일 변환)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
