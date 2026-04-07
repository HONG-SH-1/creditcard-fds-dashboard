# 제1장 1-4: 기초 통계
# - 슬라이드용(기본): report_figures/eda/ch8_descriptive_compact.png  (Time·Amount만, 읽기 쉬운 크기)
# - 부록 CSV: report_figures/eda/ch8_descriptive_statistics.csv (전체 변수 describe)
# - 선택: python scripts/report_plots/plot_ch8_descriptive_statistics.py --full  → 전체 변수 PNG (부록용, 글씨 작음)
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from fds_pipeline import CSV_DEFAULT
from report.report_figures_common import REPORT_FIGURES_EDA, setup_korean_matplotlib_font

OUT_DIR = _PROJECT_ROOT / REPORT_FIGURES_EDA
OUT_CSV_FULL = OUT_DIR / "ch8_descriptive_statistics.csv"
OUT_CSV_BY_CLASS = OUT_DIR / "ch8_descriptive_by_class_Time_Amount.csv"
OUT_PNG_COMPACT = OUT_DIR / "ch8_descriptive_compact.png"
OUT_PNG_FULL = OUT_DIR / "ch8_descriptive_statistics_full.png"

STAT_COLS_KR = {
    "count": "개수",
    "mean": "평균",
    "std": "표준편차",
    "min": "최소",
    "25%": "25%",
    "50%": "50%",
    "75%": "75%",
    "max": "최대",
}


def _fmt_cell(x: float) -> str:
    if pd.isna(x):
        return ""
    if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-4):
        return f"{x:.4e}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _build_class_time_amount_cells(df: pd.DataFrame) -> tuple[list[list[str]], list[str]]:
    cell_bc: list[list[str]] = []
    hdr = ["Class", "변수"] + [STAT_COLS_KR.get(c, c) for c in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    for cls in sorted(df["Class"].unique()):
        for var in ["Time", "Amount"]:
            s = df[df["Class"] == cls][var].describe()
            cell_bc.append(
                [str(int(cls)), var]
                + [
                    _fmt_cell(s.loc[k]) if k in s.index else _fmt_cell(float("nan"))
                    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
                ]
            )
    return cell_bc, hdr


def _build_two_var_overall_cells(df: pd.DataFrame) -> tuple[list[list[str]], list[str]]:
    """전체 행 기준 Time, Amount만 describe (2행)."""
    sub = df[["Time", "Amount"]].describe().T.round(6)
    sub.columns = [STAT_COLS_KR.get(str(c), str(c)) for c in sub.columns]
    col_labels = ["변수"] + list(sub.columns)
    cells = []
    for idx, row in sub.iterrows():
        cells.append([str(idx)] + [_fmt_cell(v) for v in row.values])
    return cells, col_labels


def plot_compact_png(df: pd.DataFrame) -> None:
    cell_bc, hdr_bc = _build_class_time_amount_cells(df)
    cell_ov, hdr_ov = _build_two_var_overall_cells(df)

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    t0 = ax0.table(cellText=cell_bc, colLabels=hdr_bc, loc="center", cellLoc="center")
    t0.auto_set_font_size(False)
    t0.set_fontsize(8)
    t0.scale(1.02, 1.35)
    ax0.set_title("Class별 Time · Amount 기초 통계", fontsize=11, pad=10)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    t1 = ax1.table(cellText=cell_ov, colLabels=hdr_ov, loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(8.5)
    t1.scale(1.02, 1.4)
    ax1.set_title("전체 데이터 기초 통계 — Time · Amount만 (Class 미구분)", fontsize=11, pad=10)

    fig.suptitle("1-4 기초 통계 요약 (원본 creditcard.csv)", fontsize=13, y=1.02)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.06)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG_COMPACT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG(슬라이드용): {OUT_PNG_COMPACT.resolve()}")


def plot_full_png(desc_wide: pd.DataFrame) -> None:
    desc_display = desc_wide.copy()
    desc_display.columns = [STAT_COLS_KR.get(c, c) for c in desc_display.columns]
    cell_all = []
    for idx, row in desc_display.iterrows():
        cell_all.append([str(idx)] + [_fmt_cell(v) for v in row.values])
    col_all = ["변수"] + list(desc_display.columns)

    fig_h = min(32.0, max(8.0, 0.32 * len(cell_all) + 2))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    fs = max(4.5, min(6.0, 100 / max(len(cell_all), 1)))
    tbl = ax.table(cellText=cell_all, colLabels=col_all, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    tbl.scale(1.0, 1.12)
    ax.set_title(
        "전체 특성 기초 통계 — 부록용 (V1~V28, Time, Amount · Class 제외)",
        fontsize=10,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG_FULL, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG(부록용 전체): {OUT_PNG_FULL.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="V1~V28 포함 전체 변수 describe를 PNG로도 저장 (부록, 글씨 작음)",
    )
    args = parser.parse_args()

    setup_korean_matplotlib_font()

    csv_path = Path(CSV_DEFAULT)
    if not csv_path.is_file():
        raise FileNotFoundError(f"데이터가 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    feat = df.drop(columns=["Class"])
    desc = feat.describe().T.round(6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(OUT_CSV_FULL, encoding="utf-8-sig")

    rows_bc = []
    for cls in sorted(df["Class"].unique()):
        sub = df[df["Class"] == cls][["Time", "Amount"]].describe()
        for var in ["Time", "Amount"]:
            stat_row = {"Class": int(cls), "변수": var}
            for c in sub.index:
                stat_row[STAT_COLS_KR.get(str(c), str(c))] = sub.loc[c, var]
            rows_bc.append(stat_row)
    pd.DataFrame(rows_bc).to_csv(OUT_CSV_BY_CLASS, encoding="utf-8-sig", index=False)

    print(f"CSV(전체 변수): {OUT_CSV_FULL.resolve()}")
    print(f"CSV(Class별 T/A): {OUT_CSV_BY_CLASS.resolve()}")

    plot_compact_png(df)

    if args.full:
        setup_korean_matplotlib_font()
        plot_full_png(desc)


if __name__ == "__main__":
    main()
