# 동일 Train(SMOTETomek)·Test(원 불균형)에서 LR·RF·XGB 벤치마크. CV는 F2.
# python scripts/model_comparison.py [--out benchmark_results.csv]
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fds_pipeline import (
    CSV_DEFAULT,
    RANDOM_STATE,
    make_xgb_baseline,
    preprocess_creditcard_dataframe,
    resample_train_smotetomek,
    stratified_train_test_split_creditcard,
)

F2_SCORER = make_scorer(fbeta_score, beta=2)


def _load_train_test_resampled(
    csv_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X, y, _ = preprocess_creditcard_dataframe(df)
    X_train, X_test, y_train, y_test = stratified_train_test_split_creditcard(X, y)
    X_tr, y_tr = resample_train_smotetomek(X_train, y_train)
    return X_tr, y_tr, X_test, y_test


def _metrics_at_threshold(y_true, y_prob, thr: float = 0.5) -> dict[str, float]:
    y_hat = (y_prob >= thr).astype(int)
    return {
        "recall": float(recall_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat)),
        "f2": float(fbeta_score(y_true, y_hat, beta=2)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _baseline_xgb() -> xgb.XGBClassifier:
    return make_xgb_baseline()


def _baseline_rf() -> RandomForestClassifier:
    # RF 베이스라인 고정 하이퍼.
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)


def run_search_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, RandomizedSearchCV]:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "lr__C": loguniform(1e-2, 1e2),
            "lr__class_weight": [None, "balanced"],
        },
        n_iter=14,
        scoring=F2_SCORER,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search


def run_search_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[xgb.XGBClassifier, RandomizedSearchCV]:
    base = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        base_score=0.5,
        tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions={
            "n_estimators": randint(80, 320),
            "max_depth": randint(3, 11),
            "learning_rate": uniform(0.04, 0.16),
            "subsample": uniform(0.65, 0.3),
            "colsample_bytree": uniform(0.65, 0.3),
            "min_child_weight": randint(1, 9),
            "reg_lambda": loguniform(1e-3, 10),
        },
        n_iter=18,
        scoring=F2_SCORER,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search


def _flatten_params(name: str, best: Any) -> dict[str, Any]:
    if isinstance(best, Pipeline):
        params = best.named_steps["lr"].get_params()
        return {f"{name}__{k}": v for k, v in sorted(params.items()) if k in ("C", "class_weight", "solver")}
    params = best.get_params()
    keys = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
    )
    return {f"{name}__{k}": params.get(k) for k in keys}


def run_benchmark(csv_path: Path | str = CSV_DEFAULT) -> tuple[pd.DataFrame, dict[str, Any]]:
    # 임계값 0.5 기준 지표 표 + 튜닝 메타.
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} 없음. Kaggle CSV를 dataset/에 두세요.")

    X_tr, y_tr, X_te, y_te = _load_train_test_resampled(path)

    rows: list[dict[str, Any]] = []

    # --- Logistic Regression ---
    lr_base = Pipeline(
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
    lr_base.fit(X_tr, y_tr)
    m = _metrics_at_threshold(y_te, lr_base.predict_proba(X_te)[:, 1])
    rows.append({"model": "LogisticRegression (baseline)", **m, "cv_best_f2": np.nan})

    lr_best, lr_search = run_search_lr(X_tr, y_tr)
    m = _metrics_at_threshold(y_te, lr_best.predict_proba(X_te)[:, 1])
    rows.append(
        {
            "model": "LogisticRegression (tuned, CV=F2)",
            **m,
            "cv_best_f2": float(lr_search.best_score_),
        }
    )

    # --- RandomForest (2-Track, 고정 하이퍼파라미터) ---
    rf_base = _baseline_rf()
    rf_base.fit(X_tr, y_tr)
    m = _metrics_at_threshold(y_te, rf_base.predict_proba(X_te)[:, 1])
    rows.append({"model": "RandomForest (baseline, 2-Track)", **m, "cv_best_f2": np.nan})

    # --- XGBoost ---
    xgb_base = _baseline_xgb()
    xgb_base.fit(X_tr, y_tr)
    m = _metrics_at_threshold(y_te, xgb_base.predict_proba(X_te)[:, 1])
    rows.append({"model": "XGBoost (baseline, fds defaults)", **m, "cv_best_f2": np.nan})

    xgb_best, xgb_search = run_search_xgb(X_tr, y_tr)
    m = _metrics_at_threshold(y_te, xgb_best.predict_proba(X_te)[:, 1])
    rows.append(
        {
            "model": "XGBoost (tuned, CV=F2)",
            **m,
            "cv_best_f2": float(xgb_search.best_score_),
        }
    )

    df = pd.DataFrame(rows)
    meta: dict[str, Any] = {
        "csv_path": str(path.resolve()),
        "random_state": RANDOM_STATE,
        "threshold": 0.5,
        "split": "stratified 0.2 test, SMOTETomek on train only",
        "lr_tuned_params": _flatten_params("lr", lr_best),
        "xgb_tuned_params": _flatten_params("xgb", xgb_best),
        "lr_cv_best_f2": float(lr_search.best_score_),
        "xgb_cv_best_f2": float(xgb_search.best_score_),
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold-out 벤치마크 (SMOTETomek train)")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument("--out", type=Path, default=None, help="지표 CSV 저장 경로")
    args = parser.parse_args()

    df, meta = run_benchmark(args.csv)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== Hold-out Test (threshold=0.5), Train=SMOTETomek / Test=원 불균형 ===\n")
    print(df.to_string(index=False))
    print("\n=== Tuned best params (요약) ===\n")
    print("LogisticRegression:", meta["lr_tuned_params"])
    print("XGBoost:", meta["xgb_tuned_params"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n표 저장: {args.out.resolve()}")


if __name__ == "__main__":
    main()
