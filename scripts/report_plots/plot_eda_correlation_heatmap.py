# EDA 2단계: 변수 간·Class 상관 히트맵 → report_figures/eda/eda_correlation_heatmap.png
# 실행: 프로젝트 루트에서  python scripts/report_plots/plot_eda_correlation_heatmap.py
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
OUT_PATH = OUT_DIR / "eda_correlation_heatmap.png"


def main() -> None:
    setup_korean_matplotlib_font()

    csv_path = Path(CSV_DEFAULT)
    if not csv_path.is_file():
        raise FileNotFoundError(f"데이터가 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    corr = df.corr()

    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("상관계수")

    ax.set_title("변수 간 상관계수 히트맵 (원본 creditcard.csv)", fontsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"저장: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
