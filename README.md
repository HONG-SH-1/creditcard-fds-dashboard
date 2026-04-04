# 신용카드 사기 탐지(FDS) — Streamlit

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)로 이진 분류·시연 대시보드. Train만 SMOTETomek, Test는 원 불균형, 모델은 XGBoost 하나. Streamlit에서 임계값·Recall·F2·PR-AUC·SHAP 워터폴.

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, Streamlit, matplotlib, joblib.

---

## 데이터

| 파일 | 설명 |
|------|------|
| `dataset/creditcard.csv` | 학습용. 레포에 넣지 않음. Kaggle에서 받아 둔다. |
| `dataset/creditcard_sample_100.xlsx` | 100행 샘플. 파이프라인 기본 입력은 CSV. |

Kaggle 페이지 조건을 따른다.

---

## 준비

Python 3.10+ 권장.

```bash
pip install -r requirements.txt
```

`dataset/creditcard.csv` 존재 확인.

---

## 실행

### 대시보드

```bash
streamlit run app.py
```

`artifacts/`에 `fds_model.pkl`, `fds_scaler.pkl`, `fds_X_test.pkl`, `fds_y_test.pkl`가 있으면 로드만 한다. 없으면 시작 시 학습. 시연 전에는 아래로 미리 만들어 두는 편이 낫다.

```bash
python scripts/train_save_artifacts.py
```

생성물은 `.gitignore`에 둔다.

### 보고서용 그림

`report_figures/*.png`는 Git에 넣지 않는다. 로컬에서 생성.

```bash
python report/generate_report_extras.py
python report/generate_report_model_diagrams.py
```

`fds_report_figures.ipynb`

### Excel 샘플 (선택)

전체 CSV가 있을 때.

```bash
pip install openpyxl
python scripts/make_creditcard_sample_excel.py
```

### 모델 비교 (선택)

`scripts/model_comparison.py`: Train(SMOTETomek)·Test 동일 분할에서 LR·RF·XGB와 LR·XGB 튜닝(RandomizedSearchCV, CV=F2). Test 지표: Recall, F1, F2, PR-AUC(임계값 0.5).

```bash
python scripts/model_comparison.py
python scripts/model_comparison.py --out benchmark_results.csv
```

### 보고서용 표·메타 (선택)

```bash
python report/export_report_bundle.py
```

`report_outputs/`에 CSV·MD·JSON. 로컬 생성물은 `.gitignore`.

- `docs/f1_f2_for_report.md` — F1/F2 문장 초안
- `docs/shap_terminology_for_report.md` — SHAP 표현 정리

---

## 지표

불균형이라 Accuracy만 쓰기 어렵다. F1은 Precision·Recall을 같은 비중(β=1), F2(β=2)는 Recall에 더 무게. 튜닝 CV와 Streamlit 사이드바는 F2. PR-AUC는 참고.

---

## 동작 요약

- 전처리: `Time` → 하루 주기 sin/cos 후 `Time` 제거, `Amount` → `RobustScaler`(전체 df fit — `fds_pipeline.py`).
- 분할: 층화 8:2, `random_state=42`. Train에만 SMOTETomek.
- 모델: `fds_pipeline.train_xgb_pipeline`의 고정 하이퍼 XGB.
- 대시보드: 임계값 변경 시 Recall·F2 재계산, PR-AUC는 고정. SHAP은 확률 기반 Explainer + `shap_waterfall_style.py`.

---

## 디렉터리

| 경로 | 역할 |
|------|------|
| `app.py` | Streamlit |
| `fds_pipeline.py` | 전처리·분할·리샘플·학습 |
| `shap_waterfall_style.py` | SHAP 워터폴 보정 |
| `scripts/train_save_artifacts.py` | pkl 저장 |
| `scripts/model_comparison.py` | 벤치마크 |
| `report/export_report_bundle.py` | 보고서용 표·메타 |
| `report/generate_report_*.py` | 보고서용 PNG |
| `docs/` | 보고서용 메모 |

---

## 라이선스

코드는 자유롭게 쓰면 된다. 데이터는 Kaggle 조건을 따른다.
