# RobustScaler full vs train-only, SMOTETomek vs scale_pos_weight — 동일 split·동일 XGB 베이스라인.
# python scripts/compare_preprocessing_strategies.py [--csv path]
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from fds_pipeline import (
    CSV_DEFAULT,
    RANDOM_STATE,
    SECONDS_PER_DAY,
    TEST_SIZE,
    XGB_BASELINE_PARAMS,
    preprocess_creditcard_dataframe,
    resample_train_smotetomek,
)

THR = 0.5


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = THR) -> dict[str, float]:
    y_hat = (y_prob >= thr).astype(int)
    return {
        "recall": float(recall_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat)),
        "f2": float(fbeta_score(y_true, y_hat, beta=2)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _time_features_amount_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    out["Time_sin"] = np.sin(2 * np.pi * out["Time"] / SECONDS_PER_DAY)
    out["Time_cos"] = np.cos(2 * np.pi * out["Time"] / SECONDS_PER_DAY)
    out = out.drop(["Time"], axis=1)
    y = out["Class"]
    X = out.drop("Class", axis=1)
    return X, y


def _same_split_indices(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    return train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass


def run_experiments(csv_path: Path, *, max_rows: int | None = None) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} 없음.")

    df = pd.read_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
        print(f"(앞 {max_rows:,}행만 사용. 전체 데이터는 SMOTETomek 때문에 수 분~수십 분 걸릴 수 있음)\n")
    X_time_raw, y = _time_features_amount_raw(df)
    X_full, y_check, _ = preprocess_creditcard_dataframe(df)
    assert np.array_equal(y.values, y_check.values)

    idx_train, idx_test = _same_split_indices(y)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    # --- A: RobustScaler full df fit (현재 fds_pipeline과 동일) ---
    X_train_full = X_full.iloc[idx_train].reset_index(drop=True)
    X_test_full = X_full.iloc[idx_test].reset_index(drop=True)

    # --- B: RobustScaler train-only fit on Amount ---
    X_tr_raw = X_time_raw.iloc[idx_train].reset_index(drop=True)
    X_te_raw = X_time_raw.iloc[idx_test].reset_index(drop=True)
    scaler_tr = RobustScaler()
    scaler_tr.fit(X_tr_raw["Amount"].values.reshape(-1, 1))
    X_train_to = X_tr_raw.copy()
    X_test_to = X_te_raw.copy()
    X_train_to["Amount"] = scaler_tr.transform(X_tr_raw["Amount"].values.reshape(-1, 1))
    X_test_to["Amount"] = scaler_tr.transform(X_te_raw["Amount"].values.reshape(-1, 1))

    rows_scale: list[dict[str, str | float]] = []
    for i, (name, Xtr, Xte) in enumerate(
        [
            ("RobustScaler full df fit (현재 파이프라인)", X_train_full, X_test_full),
            ("RobustScaler train-only fit", X_train_to, X_test_to),
        ],
        start=1,
    ):
        print(f"[실험 1 - {i}/2] SMOTETomek + 학습: {name[:40]}...", flush=True)
        Xtr_r, ytr_r = resample_train_smotetomek(Xtr, y_train)
        model = xgb.XGBClassifier(**XGB_BASELINE_PARAMS)
        model.fit(Xtr_r, ytr_r)
        prob = model.predict_proba(Xte)[:, 1]
        m = _metrics(y_test.values, prob)
        rows_scale.append(
            {
                "스케일 방식": name,
                **{k: round(v, 6) for k, v in m.items()},
            }
        )

    # --- C: SMOTETomek vs scale_pos_weight (스케일은 full-fit으로 통일해 불균형 전략만 비교) ---
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    print("[실험 2 - 1/2] SMOTETomek + 학습...", flush=True)
    Xtr_r, ytr_r = resample_train_smotetomek(X_train_full, y_train)
    m1 = xgb.XGBClassifier(**XGB_BASELINE_PARAMS)
    m1.fit(Xtr_r, ytr_r)
    prob1 = m1.predict_proba(X_test_full)[:, 1]
    row_smote = {"불균형 처리": f"SMOTETomek (train 사기 {pos}건 → 리샘플 후 학습)", **_metrics(y_test.values, prob1)}

    print("[실험 2 - 2/2] scale_pos_weight만 학습...", flush=True)
    params_w = {**XGB_BASELINE_PARAMS, "scale_pos_weight": spw}
    m2 = xgb.XGBClassifier(**params_w)
    m2.fit(X_train_full, y_train)
    prob2 = m2.predict_proba(X_test_full)[:, 1]
    row_weight = {
        "불균형 처리": f"리샘플 없음 + scale_pos_weight={spw:.4f} (neg/pos)",
        **_metrics(y_test.values, prob2),
    }

    print("\n=== 동일 split (stratify, random_state=%s, test_size=%s), 동일 XGB 베이스라인 ===\n" % (RANDOM_STATE, TEST_SIZE))
    print("Train/Test 건수: Train=%d (사기 %d) | Test=%d (사기 %d)\n" % (len(y_train), pos, len(y_test), int((y_test == 1).sum())))

    df_scale = pd.DataFrame(rows_scale)
    print("[표] 실험 1: Amount 스케일만 다름. 둘 다 Train에 SMOTETomek 후 학습\n")
    print(df_scale.to_string(index=False))
    print()

    df_imb = pd.DataFrame([row_smote, row_weight])
    print("[표] 실험 2: RobustScaler는 full df fit로 통일. 불균형 처리만 다름\n")
    print(df_imb.to_string(index=False))
    print("\n임계값 %.1f, 지표는 recall, f1, f2, pr_auc(test)." % THR)
    print("docs/leakage_and_scaling.md 와 함께 보면 설명과 맞물립니다.\n")


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="전처리 전략 비교 (동일 split)")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="앞 N행만 사용(빠른 확인용). 생략 시 전체 행(느릴 수 있음).",
    )
    args = parser.parse_args()
    run_experiments(args.csv, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
