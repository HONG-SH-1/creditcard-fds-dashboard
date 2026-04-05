# LR·RF·XGB 베이스라인 + LR·XGB 튜닝 5종 동일 split 비교.
# ROC/PR/막대/혼동행렬/보정곡선/임계값-Recall·Precision·F1 곡선 + CSV.
# python scripts/model_comparison_visualize.py [--csv path] [--out-dir report_figures]
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REPORT = _ROOT / "report"
for p in (_ROOT, _REPORT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fds_pipeline import CSV_DEFAULT, RANDOM_STATE, make_xgb_baseline

_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from model_comparison import (
    _baseline_rf,
    _load_train_test_resampled,
    _metrics_at_threshold,
    run_search_lr,
    run_search_xgb,
)

from report_figures_common import REPORT_FIGURES_OUT, setup_korean_matplotlib_font

THR = 0.5

# (dict_key, 한글 표시명)
MODEL_ORDER: list[tuple[str, str]] = [
    ("lr_base", "로지스틱 회귀 (baseline)"),
    ("lr_tuned", "로지스틱 회귀 (tuned, CV=F2)"),
    ("rf_base", "랜덤 포레스트 (baseline)"),
    ("xgb_base", "XGBoost (baseline)"),
    ("xgb_tuned", "XGBoost (tuned, CV=F2)"),
]

COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756"]


def _make_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_all_models_and_probs(
    csv_path: Path,
) -> tuple[dict[str, np.ndarray], pd.Series, dict[str, float]]:
    X_tr, y_tr, X_te, y_te = _load_train_test_resampled(csv_path)
    probs: dict[str, np.ndarray] = {}

    print("[1/5] 로지스틱 baseline...", flush=True)
    lr = _make_lr_pipeline()
    lr.fit(X_tr, y_tr)
    probs["lr_base"] = lr.predict_proba(X_te)[:, 1]

    print("[2/5] 로지스틱 RandomizedSearchCV (F2)...", flush=True)
    lr_best, lr_search = run_search_lr(X_tr, y_tr)
    probs["lr_tuned"] = lr_best.predict_proba(X_te)[:, 1]

    print("[3/5] 랜덤 포레스트 baseline...", flush=True)
    rf = _baseline_rf()
    rf.fit(X_tr, y_tr)
    probs["rf_base"] = rf.predict_proba(X_te)[:, 1]

    print("[4/5] XGBoost baseline...", flush=True)
    xgb_b = make_xgb_baseline()
    xgb_b.fit(X_tr, y_tr)
    probs["xgb_base"] = xgb_b.predict_proba(X_te)[:, 1]

    print("[5/5] XGBoost RandomizedSearchCV (F2)...", flush=True)
    xgb_best, xgb_search = run_search_xgb(X_tr, y_tr)
    probs["xgb_tuned"] = xgb_best.predict_proba(X_te)[:, 1]

    meta = {
        "lr_cv_best_f2": float(lr_search.best_score_),
        "xgb_cv_best_f2": float(xgb_search.best_score_),
    }
    return probs, y_te, meta


def metrics_table(
    y_true: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    meta: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for key, label in MODEL_ORDER:
        p = prob_dict[key]
        m = _metrics_at_threshold(y_true, p, thr=THR)
        fpr, tpr, _ = roc_curve(y_true, p)
        m["roc_auc"] = float(auc(fpr, tpr))
        cv = np.nan
        if key == "lr_tuned":
            cv = meta["lr_cv_best_f2"]
        elif key == "xgb_tuned":
            cv = meta["xgb_cv_best_f2"]
        rows.append({"model": label, "cv_best_f2_train": cv, **m})
    return pd.DataFrame(rows)


def plot_metrics_bar(df: pd.DataFrame, out: Path) -> None:
    setup_korean_matplotlib_font()
    metrics = ["recall", "f1", "f2", "pr_auc", "roc_auc"]
    labels_k = ["Recall", "F1", "F2", "PR-AUC", "ROC-AUC"]
    x = np.arange(len(metrics))
    n = len(MODEL_ORDER)
    w = min(0.8 / n, 0.14)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for i, ((key, name_k), c) in enumerate(zip(MODEL_ORDER, COLORS)):
        vals = [df.loc[df["model"] == name_k, m].values[0] for m in metrics]
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, vals, width=w, label=name_k, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_k)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("점수")
    ax.set_title(
        f"Hold-out 지표 비교 (임계값 {THR}) · Train=SMOTETomek / Test=원 불균형"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", out.resolve())


def plot_roc(y_true: np.ndarray, prob_dict: dict[str, np.ndarray], out: Path) -> None:
    setup_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for (key, name_k), color in zip(MODEL_ORDER, COLORS):
        fpr, tpr, _ = roc_curve(y_true, prob_dict[key])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name_k} (AUC={a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC 곡선 (Test)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", out.resolve())


def plot_pr(y_true: np.ndarray, prob_dict: dict[str, np.ndarray], out: Path) -> None:
    setup_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for (key, name_k), color in zip(MODEL_ORDER, COLORS):
        prec, rec, _ = precision_recall_curve(y_true, prob_dict[key])
        ap = average_precision_score(y_true, prob_dict[key])
        ax.plot(rec, prec, color=color, lw=2, label=f"{name_k} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall 곡선 (Test)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", out.resolve())


def plot_confusion_grids(
    y_true: np.ndarray, prob_dict: dict[str, np.ndarray], out: Path
) -> None:
    setup_korean_matplotlib_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()
    for idx, ((key, name_k), color) in enumerate(zip(MODEL_ORDER, COLORS)):
        ax = axes_flat[idx]
        y_hat = (prob_dict[key] >= THR).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_hat,
            ax=ax,
            colorbar=False,
            labels=[0, 1],
        )
        ax.set_title(f"{name_k}\n(임계값 {THR})", fontsize=9)
    axes_flat[5].set_visible(False)
    plt.suptitle("혼동 행렬 (Test, 정상=0 / 사기=1)", y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", out.resolve())


def plot_calibration(
    y_true: np.ndarray, prob_dict: dict[str, np.ndarray], out: Path
) -> None:
    setup_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for (key, name_k), color in zip(MODEL_ORDER, COLORS):
        prob_pos = prob_dict[key]
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, prob_pos, n_bins=10, strategy="quantile"
            )
        except ValueError:
            frac_pos, mean_pred = calibration_curve(
                y_true, prob_pos, n_bins=10, strategy="uniform"
            )
        ax.plot(mean_pred, frac_pos, "s-", color=color, lw=2, label=name_k)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="완전 보정")
    ax.set_xlabel("평균 예측 확률 (빈)")
    ax.set_ylabel("양성 비율 (실제)")
    ax.set_title("보정 곡선 (Test, calibration_curve)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", out.resolve())


def plot_threshold_curves(
    y_true: np.ndarray, prob_dict: dict[str, np.ndarray], out: Path
) -> None:
    setup_korean_matplotlib_font()
    thresholds = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for (key, name_k), color in zip(MODEL_ORDER, COLORS):
        recs, precs, f1s = [], [], []
        for t in thresholds:
            yh = (prob_dict[key] >= t).astype(int)
            recs.append(float(recall_score(y_true, yh, zero_division=0)))
            precs.append(float(precision_score(y_true, yh, zero_division=0)))
            f1s.append(float(f1_score(y_true, yh, zero_division=0)))
        axes[0].plot(thresholds, recs, color=color, lw=2, label=name_k)
        axes[1].plot(thresholds, precs, color=color, lw=2, label=name_k)
        axes[2].plot(thresholds, f1s, color=color, lw=2, label=name_k)
    axes[0].set_title("Recall vs 임계값")
    axes[1].set_title("Precision vs 임계값")
    axes[2].set_title("F1 vs 임계값")
    for ax in axes:
        ax.set_xlabel("임계값")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=6)
    axes[0].set_ylabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[2].set_ylabel("F1")
    plt.suptitle("임계값에 따른 지표 (Test)", y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", out.resolve())


def run_all(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} 없음.")

    out_dir.mkdir(parents=True, exist_ok=True)
    probs, y_te, meta = fit_all_models_and_probs(csv_path)
    y_true = y_te.values

    df = metrics_table(y_true, probs, meta)
    csv_out = out_dir / "model_compare_all_metrics.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print("저장:", csv_out.resolve())

    plot_metrics_bar(df, out_dir / "model_compare_all_metrics_bar.png")
    plot_roc(y_true, probs, out_dir / "model_compare_all_roc.png")
    plot_pr(y_true, probs, out_dir / "model_compare_all_pr.png")
    plot_confusion_grids(y_true, probs, out_dir / "model_compare_all_confusion.png")
    plot_calibration(y_true, probs, out_dir / "model_compare_all_calibration.png")
    plot_threshold_curves(y_true, probs, out_dir / "model_compare_all_threshold_curves.png")

    print("\n표 요약:\n", df.to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LR·RF·XGB 전체 비교 그림 (5모델 + 보정 + 임계값 곡선)"
    )
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=REPORT_FIGURES_OUT)
    args = parser.parse_args()
    run_all(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
