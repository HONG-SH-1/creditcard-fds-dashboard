# 제5장 5-7 슬라이드 선택용: 검증(val)에서 임계값별 F2 곡선 (2차 실험과 동일 분할·모델)
# model_comparison_round2.py 와 동일한 학습·검증 확률로 F2(threshold) 전체 스캔 시각화.
# 출력: report_figures/eda/round2_val_f2_vs_threshold.png
# 실행: 프로젝트 루트에서  python scripts/report_plots/plot_round2_val_f2_vs_threshold.py
# 주의: RF·XGB·XGB튜닝 학습으로 수 분 소요될 수 있음.
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import repro_threads  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import fbeta_score

from fds_pipeline import CSV_DEFAULT, make_xgb_baseline

from model_comparison import (
    _baseline_rf,
    _load_fit_val_test_resampled,
    run_search_xgb,
)

from report.report_figures_common import REPORT_FIGURES_EDA, setup_korean_matplotlib_font

OUT_DIR = _ROOT / REPORT_FIGURES_EDA
OUT_PATH = OUT_DIR / "round2_val_f2_vs_threshold.png"

VAL_FRACTION = 0.15
THRS = np.linspace(0.01, 0.99, 99)
LABELS = [
    ("RF 베이스", "#4C78A8"),
    ("XGB 베이스", "#F58518"),
    ("XGB 튜닝 (CV=F2)", "#E45756"),
]


def _f2_curve(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    out = []
    for th in THRS:
        y_hat = (y_prob >= th).astype(int)
        out.append(float(fbeta_score(y_true, y_hat, beta=2, zero_division=0)))
    return np.array(out)


def main() -> None:
    setup_korean_matplotlib_font()

    path = Path(CSV_DEFAULT)
    if not path.is_file():
        raise FileNotFoundError(f"{path} 없음.")

    X_res, y_res, X_val, y_val, _, _ = _load_fit_val_test_resampled(path, VAL_FRACTION)

    print("[1/3] RandomForest baseline...", flush=True)
    rf = _baseline_rf()
    rf.fit(X_res, y_res)
    p_rf = rf.predict_proba(X_val)[:, 1]

    print("[2/3] XGBoost baseline...", flush=True)
    xgb_b = make_xgb_baseline()
    xgb_b.fit(X_res, y_res)
    p_xb = xgb_b.predict_proba(X_val)[:, 1]

    print("[3/3] XGBoost RandomizedSearchCV...", flush=True)
    xgb_best, _ = run_search_xgb(X_res, y_res)
    p_xt = xgb_best.predict_proba(X_val)[:, 1]

    yv = y_val.values
    curves = [
        _f2_curve(yv, p_rf),
        _f2_curve(yv, p_xb),
        _f2_curve(yv, p_xt),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, color), f2s in zip(LABELS, curves):
        ax.plot(THRS, f2s, lw=2, label=name, color=color)
    ax.set_xlabel("임계값 (threshold)")
    ax.set_ylabel("F2 (검증 세트)")
    ax.set_title(
        f"검증(val)에서 임계값별 F2 — 2차 실험과 동일 분할 (val_fraction={VAL_FRACTION})"
    )
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
