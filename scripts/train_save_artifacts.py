# 학습 후 artifacts/*.pkl 저장. app은 이 파일이 있으면 로드만 함.
# python scripts/train_save_artifacts.py
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import repro_threads  # noqa: E402

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
