# EDA 1단계: Class 건수 막대그래프 → report_figures/eda/eda_class_distribution.png
# 실행: 프로젝트 루트에서  python scripts/report_plots/plot_eda_class_distribution.py
from __future__ import annotations

import sys
from pathlib import Path

# scripts/report_plots/ 기준 프로젝트 루트 = 상위 2단계
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from fds_pipeline import CSV_DEFAULT
from report.report_figures_common import REPORT_FIGURES_EDA, setup_korean_matplotlib_font

OUT_DIR = _PROJECT_ROOT / REPORT_FIGURES_EDA
OUT_PATH = OUT_DIR / "eda_class_distribution.png"


def main() -> None:
    setup_korean_matplotlib_font()

    csv_path = Path(CSV_DEFAULT)
    if not csv_path.is_file():
        raise FileNotFoundError(f"데이터가 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    counts = df["Class"].value_counts().sort_index()
    label_map = {0: "정상 (0)", 1: "사기 (1)"}
    labels = [label_map[int(i)] for i in counts.index]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, counts.values, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("건수")
    ax.set_title("Class 분포 (원본 creditcard.csv)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"저장: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
