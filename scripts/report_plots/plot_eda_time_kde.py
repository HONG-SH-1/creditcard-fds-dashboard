# EDA 3단계: Time KDE → report_figures/eda/eda_time_kde.png
# 실행: 프로젝트 루트에서  python scripts/report_plots/plot_eda_time_kde.py
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fds_pipeline import CSV_DEFAULT
from report.report_figures_common import REPORT_FIGURES_EDA, setup_korean_matplotlib_font

OUT_DIR = _PROJECT_ROOT / REPORT_FIGURES_EDA
OUT_PATH = OUT_DIR / "eda_time_kde.png"


def main() -> None:
    setup_korean_matplotlib_font()

    csv_path = Path(CSV_DEFAULT)
    if not csv_path.is_file():
        raise FileNotFoundError(f"데이터가 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    t = df["Time"].astype(float).values

    fig, ax = plt.subplots(figsize=(7, 4))
    kde = stats.gaussian_kde(t)
    xs = np.linspace(t.min(), t.max(), 500)
    ax.plot(xs, kde(xs), color="#1f77b4", lw=1.8)
    ax.axvline(86_400, color="coral", ls="--", lw=1.2, label="86,400초 (1일)")
    ax.axvline(172_800, color="seagreen", ls="--", lw=1.2, label="172,800초 (2일)")
    ax.set_xlabel("Time (첫 거래 이후 경과 초)")
    ax.set_ylabel("밀도")
    ax.set_title("Time 분포 KDE (원본 creditcard.csv)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"저장: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
