# creditcard 전처리·XGB 학습. app·스크립트에서 동일하게 import.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import repro_threads  # noqa: E402 — numpy 이전에 BLAS 스레드 고정

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

CSV_DEFAULT = Path("dataset/creditcard.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
SECONDS_PER_DAY = 86400

# 베이스라인 XGB (model_comparison·보고서와 동일 값)
XGB_BASELINE_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,  # 재현성(병렬 트리 순서 비결정성 방지). 속도는 느려질 수 있음.
    "base_score": 0.5,
}


def make_xgb_baseline() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**XGB_BASELINE_PARAMS)


def preprocess_creditcard_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, RobustScaler]:
    # Time → sin/cos(86400초 기준) 후 Time 제거. Amount는 RobustScaler로 변환(df 전체에 fit).
    # 분할·SMOTE는 이 함수 밖에서 한다. EDA 원시 분포는 입력 df로 그린다.
    out = df.copy()
    out["Time_sin"] = np.sin(2 * np.pi * out["Time"] / SECONDS_PER_DAY)
    out["Time_cos"] = np.cos(2 * np.pi * out["Time"] / SECONDS_PER_DAY)
    out = out.drop(["Time"], axis=1)
    scaler = RobustScaler()
    out["Amount"] = scaler.fit_transform(out["Amount"].values.reshape(-1, 1))
    X = out.drop("Class", axis=1)
    y = out["Class"]
    return X, y, scaler


def stratified_train_test_split_creditcard(
    X: pd.DataFrame, y: pd.Series, *, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def resample_train_smotetomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    smt = SMOTETomek(random_state=random_state)
    return smt.fit_resample(X_train, y_train)


def train_xgb_pipeline(
    csv_path: Path | str = CSV_DEFAULT,
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series, RobustScaler]:
    # CSV 읽기 → 전처리 → 층화 분할 → Train에 SMOTETomek → XGBoost fit. 반환: model, X_test, y_test, scaler.
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} 가 없습니다. Kaggle 등에서 받아 dataset 폴더에 두세요."
        )

    df = pd.read_csv(path)
    X, y, scaler = preprocess_creditcard_dataframe(df)
    X_train, X_test, y_train, y_test = stratified_train_test_split_creditcard(X, y)
    X_train_res, y_train_res = resample_train_smotetomek(X_train, y_train)

    model = make_xgb_baseline()
    model.fit(X_train_res, y_train_res)

    return model, X_test, y_test, scaler
