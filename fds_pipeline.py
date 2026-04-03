"""
FDS 학습 파이프라인 (단일 진실 소스).
app.py · train_save_artifacts.py · 보고서 스크립트에서 동일 전처리를 재사용합니다.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

CSV_DEFAULT = Path("dataset/creditcard.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def preprocess_creditcard_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, RobustScaler]:
    """원시 creditcard 행( Time, Amount, Class, V* … ) → 특성 행렬 X, 정답 y, Amount용 스케일러.

    학습 파이프라인과 동일: Time → sin/cos, Time 컬럼 제거, Amount는 **전체 df**에 대해 RobustScaler fit 후 변환,
    그 다음 ``stratified_train_test_split_creditcard`` 로 나눈다.

    보고서 전용 시각화(EDA KDE 등)는 원시 ``df``를 따로 쓰면 된다.
    """
    out = df.copy()
    out["Time_sin"] = np.sin(2 * np.pi * out["Time"] / 86400)
    out["Time_cos"] = np.cos(2 * np.pi * out["Time"] / 86400)
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
    """
    creditcard CSV를 읽어 전처리(pandas) → SMOTETomek → XGBoost 학습까지 수행.

    Returns
    -------
    model, X_test, y_test, scaler
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} 가 없습니다. Kaggle 등에서 받아 dataset 폴더에 두세요."
        )

    df = pd.read_csv(path)
    X, y, scaler = preprocess_creditcard_dataframe(df)
    X_train, X_test, y_train, y_test = stratified_train_test_split_creditcard(X, y)
    X_train_res, y_train_res = resample_train_smotetomek(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        base_score=0.5,
    )
    model.fit(X_train_res, y_train_res)

    return model, X_test, y_test, scaler
