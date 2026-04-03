"""
로컬에서 1회 실행: 학습 후 artifacts/*.pkl 저장.
Streamlit(app.py)은 이 파일들이 있으면 학습 없이 로드만 수행합니다.
"""
from pathlib import Path

import joblib

from fds_pipeline import CSV_DEFAULT, train_xgb_pipeline

OUT_DIR = Path("artifacts")


def train_and_save() -> None:
    model, X_test, y_test, scaler = train_xgb_pipeline(CSV_DEFAULT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "fds_model.pkl")
    joblib.dump(scaler, OUT_DIR / "fds_scaler.pkl")
    joblib.dump(X_test, OUT_DIR / "fds_X_test.pkl")
    joblib.dump(y_test, OUT_DIR / "fds_y_test.pkl")

    print("저장 완료:")
    for name in ("fds_model.pkl", "fds_scaler.pkl", "fds_X_test.pkl", "fds_y_test.pkl"):
        p = OUT_DIR / name
        print(f"  {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    train_and_save()
