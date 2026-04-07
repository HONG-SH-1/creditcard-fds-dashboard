# EDA 4단계: Amount 박스플롯 + 로그 히스토그램 → report_figures/eda/eda_amount_box_hist.png
# 실행: 프로젝트 루트에서  python scripts/report_plots/plot_eda_amount.py
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fds_pipeline import CSV_DEFAULT
from report.report_figures_common import REPORT_FIGURES_EDA, setup_korean_matplotlib_font

OUT_DIR = _PROJECT_ROOT / REPORT_FIGURES_EDA
OUT_PATH = OUT_DIR / "eda_amount_box_hist.png"


def main() -> None:
    setup_korean_matplotlib_font()

    csv_path = Path(CSV_DEFAULT)
    if not csv_path.is_file():
        raise FileNotFoundError(f"데이터가 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    amt0 = df.loc[df["Class"] == 0, "Amount"].astype(float)
    amt1 = df.loc[df["Class"] == 1, "Amount"].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bp = ax1.boxplot(
        [amt0, amt1],
        tick_labels=["정상 (0)", "사기 (1)"],
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], ("#4c72b0", "#dd8452")):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax1.set_ylabel("Amount")
    ax1.set_title("Class별 Amount 박스플롯")

    log_amt = np.log10(df["Amount"].astype(float) + 1.0)
    ax2.hist(log_amt, bins=80, color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("log10(Amount + 1)")
    ax2.set_ylabel("빈도")
    ax2.set_title("Amount 분포 (로그 스케일, 전체)")

    fig.suptitle("Amount 분포 (원본 creditcard.csv)", fontsize=12, y=1.02)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
