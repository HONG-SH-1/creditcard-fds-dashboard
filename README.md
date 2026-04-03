# 신용카드 이상거래(FDS) 분류 — Streamlit 대시보드

Kaggle 신용카드 사기 탐지 벤치마크를 사용한 **이진 분류·시연용 프로토타입**입니다.  
Train에만 SMOTETomek, Hold-out Test는 원 불균형 유지 후 **XGBoost** 학습, **Streamlit**에서 임계값·지표·**SHAP 워터폴**을 확인합니다.

**Credit card fraud (FDS) prototype** — SMOTETomek on train only, XGBoost, Streamlit (Recall / F2 vs threshold, fixed PR-AUC), SHAP waterfall explanations.

---

## 기술 스택

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · SHAP · Streamlit · matplotlib · joblib

---

## 데이터

| 파일 | 설명 |
|------|------|
| **`dataset/creditcard.csv`** | **실행·학습에 필요**한 전체 데이터. 레포에는 포함하지 않음(용량·Kaggle 이용 정책). 아래 링크에서 받은 뒤 이 경로에 저장합니다. |
| **`dataset/creditcard_sample_100.xlsx`** | **레포에 포함**: 원본과 **동일 컬럼·형식**으로 **100행만** 뽑은 샘플. 스키마·컬럼 확인용이며, **그대로는 본 파이프라인 입력이 아님** (`fds_pipeline`은 CSV 경로를 사용). |

- **출처:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
  데이터 사용 조건은 Kaggle 페이지의 라이선스·규정을 따릅니다.

샘플만으로는 학습·지표가 의미 있게 나오기 어렵습니다. **보고서/재현은 전체 CSV**를 사용하세요.

---

## 사전 준비

- Python 3.10+ 권장  
- (선택) 가상환경: `python -m venv venv` 후 활성화

```bash
pip install -r requirements.txt
```

전체 CSV를 `dataset/creditcard.csv`에 두었는지 확인합니다.

---

## 실행

### 1) Streamlit 대시보드

```bash
streamlit run app.py
```

- `artifacts/`에 `fds_model.pkl`, `fds_scaler.pkl`, `fds_X_test.pkl`, `fds_y_test.pkl`가 모두 있으면 **학습 없이 로드**합니다.
- 없으면 시작 시 **전체 학습**(시간 소요). 시연 전에 아래 스크립트로 아티팩트를 만들어 두는 것을 권장합니다.

### 2) 아티팩트만 로컬에서 생성

```bash
python train_save_artifacts.py
```

생성물은 `.gitignore`로 Git에 올리지 않습니다.

### 3) 보고서용 도식 (선택)

`report_figures/*.png`는 Git에 포함하지 않습니다. 로컬에서 생성합니다.

```bash
python generate_report_extras.py
python generate_report_model_diagrams.py
```

노트북: `fds_report_figures.ipynb`

### 4) Excel 샘플 재생성 (선택)

전체 `creditcard.csv`가 있을 때만:

```bash
pip install openpyxl   # 스크립트 전용
python make_creditcard_sample_excel.py
```

---

## 동작 요약

- **전처리:** `Time` → 일 단위 sin/cos 후 원시 `Time` 제거, `Amount` → `RobustScaler` (동일 규칙은 `fds_pipeline.py` 참고).
- **분할:** 층화 8:2, `random_state=42`. **Train에만** SMOTETomek, Test 미증식.
- **모델:** XGBoost (`fds_pipeline.train_xgb_pipeline`).
- **대시보드:** 임계값에 따라 **Recall·F2** 재계산, **PR-AUC**는 고정. SHAP은 확률 출력 기반 Explainer + 워터폴 후처리(`shap_waterfall_style.py`).

---

## 저장소 구조 (요약)

| 경로 | 역할 |
|------|------|
| `app.py` | Streamlit UI |
| `fds_pipeline.py` | 전처리·분할·리샘플·학습 단일 진입 |
| `shap_waterfall_style.py` | SHAP 워터폴 matplotlib 정리 |
| `train_save_artifacts.py` | joblib 아티팩트 저장 |
| `generate_report_*.py` | 보고서용 PNG 생성 |
| `dataset/creditcard_sample_100.xlsx` | 스키마 참고용 소표본 |

---

## 라이선스

본 레포의 **코드**는 필요에 따라 사용하시면 됩니다.  
**데이터**는 Kaggle 데이터셋 페이지의 조건을 따릅니다.
