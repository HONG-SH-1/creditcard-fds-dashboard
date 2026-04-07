# report_figures/ 에 2-Track·배깅/부스팅·학습곡선 PNG.
# python report/generate_report_model_diagrams.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

_ROOT = Path(__file__).resolve().parent.parent
_REPORT = Path(__file__).resolve().parent
for p in (_ROOT, _REPORT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import repro_threads  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

from fds_pipeline import (
    CSV_DEFAULT,
    RANDOM_STATE,
    make_xgb_baseline,
    preprocess_creditcard_dataframe,
    resample_train_smotetomek,
    stratified_train_test_split_creditcard,
)
from report_figures_common import REPORT_FIGURES_OUT, setup_korean_matplotlib_font

CSV_PATH = CSV_DEFAULT
OUT = REPORT_FIGURES_OUT


def fig_two_track() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f4fc", ec="#1a5a7a"):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)

    # Row layout
    box(0.3, 3.3, 2.6, 1.2, "원본\nTrain", "#fff4e6", "#b35900")
    box(3.2, 3.3, 2.8, 1.2, "SMOTETomek\n(학습 전용)", "#ffe6ee", "#990033")
    ax.annotate("", xy=(6.2, 3.9), xytext=(6.0, 3.9), arrowprops=dict(arrowstyle="-|>", lw=1.5))
    box(6.4, 3.3, 2.4, 1.2, "증식 Train", "#e8ffe8", "#206020")
    ax.annotate("", xy=(9.0, 3.9), xytext=(8.8, 3.9), arrowprops=dict(arrowstyle="-|>", lw=1.5))

    box(3.0, 1.0, 2.4, 1.0, "RandomForest\n(Bagging)", "#dde8ff", "#2244aa")
    box(6.0, 1.0, 2.4, 1.0, "XGBoost\n(Boosting)", "#dde8ff", "#2244aa")
    box(10.2, 1.0, 2.6, 1.2, "동일 Test\n(원분포)", "#f5f5f5", "#444444")

    ax.annotate(
        "",
        xy=(4.2, 2.2),
        xytext=(4.5, 3.25),
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", lw=1.2),
    )
    ax.annotate(
        "",
        xy=(7.2, 2.2),
        xytext=(7.1, 3.25),
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", lw=1.2),
    )
    ax.annotate(
        "",
        xy=(11.5, 2.2),
        xytext=(8.5, 3.25),
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.35", lw=1.2),
    )

    fig.suptitle("2-Track 비교: 동일 전처리·증식 Train으로 RF / XGB 각각 학습 → 동일 Test 평가", fontsize=12, y=0.98)
    plt.tight_layout()
    fig.savefig(OUT / "model_diagram_two_track.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", OUT / "model_diagram_two_track.png")


def fig_bagging_boosting() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, title, subtitle, blocks, mode in [
        (
            ax1,
            "Bagging (랜덤 포레스트)",
            "병렬 트리 + 다수결 / 평균 → 분산 감소",
            [("데이터\n부트스트랩", "#c8e6c9"), ("트리\n병렬 학습", "#a5d6a7"), ("투표 / 평균", "#66bb6a")],
            "h",
        ),
        (
            ax2,
            "Boosting (XGBoost)",
            "순차 트리 + 이전 오차 보정 → 편향 감소",
            [("1차 트리", "#bbdefb"), ("잔차\n보정", "#64b5f6"), ("k차 트리\n누적", "#1976d2")],
            "h",
        ),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=8)
        n = len(blocks)
        gap = 0.4
        w = (10 - gap * (n + 1)) / n
        x0 = gap
        y = 1.2
        h = 1.4
        for i, (text, c) in enumerate(blocks):
            x = x0 + i * (w + gap)
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0.03", facecolor=c, edgecolor="#333333", linewidth=1
                )
            )
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
            if i < n - 1:
                ax.annotate(
                    "→",
                    xy=(x + w + gap / 2, y + h / 2),
                    fontsize=14,
                    ha="center",
                    va="center",
                )

    fig.suptitle("앙상블 두 축: 배깅 vs 부스팅 (과제 3.1 서술용 개념도)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "model_diagram_bagging_boosting.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", OUT / "model_diagram_bagging_boosting.png")


def _load_train_res(max_rows_after_resample: int = 20_000) -> Tuple[pd.DataFrame, pd.Series]:
    # 학습곡선용: fds_pipeline과 동일 전처리·분할 후 Train 층화 축소 → SMOTETomek (수 분 이상 방지).
    df = pd.read_csv(CSV_PATH)
    X, y, _ = preprocess_creditcard_dataframe(df)
    X_train, _, y_train, _ = stratified_train_test_split_creditcard(X, y)
    cap_in = min(35_000, len(X_train))
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=cap_in, stratify=y_train, random_state=RANDOM_STATE
    )
    Xr, yr = resample_train_smotetomek(X_train, y_train)
    cap = min(max_rows_after_resample, len(Xr))
    if len(Xr) > cap:
        Xr, _, yr, _ = train_test_split(Xr, yr, train_size=cap, stratify=yr, random_state=RANDOM_STATE)
    return Xr, yr


def fig_learning_curves() -> None:
    Xr, yr = _load_train_res(max_rows_after_resample=18_000)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    # 상대 비율(0~1) — 절대 샘플 수는 len(Xr) 이하여야 함
    train_sizes = np.linspace(0.2, 1.0, 5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, est, name in [
        (
            ax1,
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
            "RandomForest",
        ),
        (ax2, make_xgb_baseline(), "XGBoost"),
    ]:
        sizes, train_sc, val_sc = learning_curve(
            est,
            Xr,
            yr,
            train_sizes=train_sizes,
            cv=cv,
            scoring="recall",
            n_jobs=1,
        )
        train_mean = train_sc.mean(axis=1)
        val_mean = val_sc.mean(axis=1)
        ax.plot(sizes, train_mean, "o-", label="Train Recall", color="#1f77b4")
        ax.plot(sizes, val_mean, "o-", label="CV Recall (평균)", color="#ff7f0e")
        ax.set_xlabel("학습 샘플 수")
        ax.set_ylabel("Recall")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "학습 크기별 Recall (3-fold CV) — 소표본 시각화, 과적합·안정성 참고\n"
        "※ GridSearch 전체는 미실시; 고정 하이퍼파라미터 기준 곡선",
        fontsize=10,
        y=1.05,
    )
    plt.tight_layout()
    fig.savefig(OUT / "model_diagram_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", OUT / "model_diagram_learning_curve.png")


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"{CSV_PATH} 가 없습니다.")
    OUT.mkdir(parents=True, exist_ok=True)
    setup_korean_matplotlib_font()
    fig_two_track()
    fig_bagging_boosting()
    fig_learning_curves()
    print("완료.")


if __name__ == "__main__":
    main()
