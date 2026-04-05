# 신용카드 사기 탐지(FDS) — Streamlit

이 저장소는 신용카드 거래가 사기인지 아닌지를 맞추는 이진 분류 모델과, 그 결과를 보여 주는 Streamlit 대시보드를 담고 있다.

데이터 출처는 Kaggle의 [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)이다. 거래 특성은 PCA로 이미 축약된 `V1`–`V28`, 시간(`Time`), 금액(`Amount`) 등으로 구성되어 있고, 레이블은 `Class`(0=정상, 1=사기)다. 데이터 사용 조건과 인용 방식은 Kaggle 페이지를 따른다.

학습 파이프라인에서는 Train에만 불균형 보정(SMOTETomek)을 적용하고, Test는 원래 불균형 비율을 그대로 둔다. 학습용으로만 증식·정리한 뒤, 평가는 실제 분포에 가까운 Test에서 하기 위해서다. 분류기는 XGBoost 한 종류를 쓴다.

대시보드에서는 예측 확률에 대한 임계값을 바꿔 보면서 Recall, F1, F2, PR-AUC를 확인하고, SHAP 워터폴로 한 건이 왜 그렇게 분류됐는지 본다.

사용 라이브러리: Python, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, Streamlit, matplotlib, joblib.

---

## 데이터

Kaggle에서 받은 전체 데이터는 용량·라이선스 때문에 이 저장소에 넣지 않는다. 대신 아래처럼 두고 쓴다.

| 파일 | 설명 |
|------|------|
| `dataset/creditcard.csv` | 모델 학습·스크립트에 쓰는 전체 CSV. 직접 받아서 같은 경로에 둔다. |
| `dataset/creditcard_sample_100.xlsx` | 열 이름·형식만 볼 때 쓰는 100행 샘플. 학습 파이프라인의 기본 입력은 CSV다. |

---

## 준비

Python 3.10 이상을 쓰는 것을 권장한다. 가상환경은 필수는 아니다.

```bash
pip install -r requirements.txt
```

그다음 `dataset/creditcard.csv`가 있는지 확인한다. 없으면 대시보드나 스크립트가 데이터를 못 찾고 멈춘다.

---

## 실행

### 대시보드

```bash
streamlit run app.py
```

처음 실행할 때 `artifacts/` 폴더에 아래 네 파일이 있으면, 이미 학습된 모델과 테스트용 데이터를 그대로 불러 온다. 없으면 앱이 시작될 때 전체 학습을 다시 돌린다(시간이 걸릴 수 있다).

- `fds_model.pkl` — 학습된 XGBoost 모델  
- `fds_scaler.pkl` — 금액 스케일링 등에 쓴 변환기  
- `fds_X_test.pkl`, `fds_y_test.pkl` — hold-out 테스트 세트  

시연이나 보고 전에 한 번 학습해 두려면:

```bash
python scripts/train_save_artifacts.py
```

이렇게 만든 파일은 보통 Git에 올리지 않도록 `.gitignore`에 넣어 두었다.

### 보고서용 그림

PNG 그림은 용량·환경마다 달라질 수 있어서 Git에 포함하지 않는다. 필요할 때 아래를 실행해 `report_figures/`에 만든다.

```bash
python report/generate_report_extras.py
python report/generate_report_model_diagrams.py
```

노트북 `fds_report_figures.ipynb`로 같은 데이터를 다른 각도에서 그릴 수도 한다.

노트북 `fds_preprocessing_walkthrough.ipynb`는 `fds_pipeline.py`와 **같은 순서**로 전처리·분할·리샘플만 셀 단위로 본다(학습·XGB·SHAP은 범위 밖). 마지막에 `train_xgb_pipeline`만 안내한다. **동작 요약**과 같은 줄기이며, 보고서 목차(데이터 이해 → 전처리 근거 → 분할·SMOTETomek)에 맞추기 좋다.

### Excel 샘플 (선택)

전체 `creditcard.csv`가 있을 때만 의미 있다. 스프레드시트로 스키마를 보고 싶을 때 쓴다.

```bash
pip install openpyxl
python scripts/make_creditcard_sample_excel.py
```

### 모델 비교 (선택)

`scripts/model_comparison.py`는 **같은 데이터 분할**을 기준으로 여러 모델을 나란히 비교한다. Train에는 SMOTETomek을 적용한 뒤, 로지스틱·랜덤포레스트·XGB의 기본 설정과, 로지스틱·XGB를 하이퍼파라미터 탐색(RandomizedSearchCV)으로 맞춘 버전을 돌린다. 교차검증에서 맞출 점수는 사기를 놓치지 않는 쪽을 더 보려고 F2로 두었다.

테스트 세트 지표는 모두 **확률이 0.5 이상이면 사기로 본다**는 같은 임계값에서 Recall, F1, F2, PR-AUC를 적는다.

```bash
python scripts/model_comparison.py
python scripts/model_comparison.py --out benchmark_results.csv
```

**5종 모델 비교 그래프** — 베이스라인 3종 + LR·XGB 튜닝(RandomizedSearchCV, CV=F2). 동일 split·Test 평가로 `report_figures/`에 아래를 저장한다. (`*.png`는 Git에 넣지 않음.)

- `model_compare_all_metrics.csv` — Recall, F1, F2, PR-AUC, ROC-AUC, `cv_best_f2_train`(튜닝 모델만)  
- `model_compare_all_metrics_bar.png` — 지표 막대  
- `model_compare_all_roc.png`, `model_compare_all_pr.png` — ROC / PR  
- `model_compare_all_confusion.png` — 혼동행렬 5분할  
- `model_compare_all_calibration.png` — 보정 곡선  
- `model_compare_all_threshold_curves.png` — 임계값–Recall·Precision·F1  

SMOTETomek + LR/XGB 탐색 때문에 **전체 CSV 기준 수십 분** 걸릴 수 있다.

```bash
python scripts/model_comparison_visualize.py
python scripts/model_comparison_visualize.py --out-dir report_figures
```

**2차(3모델, 검증 임계값)** — 랜덤 포레스트 베이스라인·XGB 베이스라인·XGB 튜닝만 대상으로 한다. 원시 train을 fit|검증으로 층화 분할한 뒤 SMOTETomek은 fit에만 적용하고, **각 모델의 검증 확률에서 F2가 최대가 되는 임계값(0.01~0.99)** 을 고른 다음 **동일 test**에서만 Recall·F1·F2 등을 계산한다. 1차는 전체 train을 리샘플해 학습하므로 **숫자를 1차 CSV와 직접 줄줄 비교하지는 말 것**.

```bash
python scripts/model_comparison_round2.py
python scripts/model_comparison_round2.py --val-fraction 0.15 --out report_figures/model_round2_val_thr.csv
```

### 보고서용 표·메타 (선택)

```bash
python report/export_report_bundle.py
```

위 스크립트가 `report_outputs/`에 지표 표(CSV·마크다운), 튜닝 요약(JSON), 실행 시각 등을 쌓는다. 보고서에 붙일 문장 초안은 아래를 보면 된다.

- `docs/f1_f2_for_report.md` — F1·F2를 보고서에 어떻게 쓸지  
- `docs/shap_terminology_for_report.md` — SHAP 설명을 코드와 맞출 때  
- `docs/leakage_and_scaling.md` — 누출·처리 순서·Amount 스케일(full vs train-only)·불균형 전략 정리  

### 전처리 비교 실험 (선택)

같은 Train/Test 분할·같은 XGB 베이스라인으로 아래를 한 번에 출력한다.

- **실험 1:** `RobustScaler`를 전체 df에 fit한 경우 vs Train에만 fit한 경우(둘 다 Train에 SMOTETomek 후 학습).  
- **실험 2:** 스케일은 전체 fit로 통일한 뒤, SMOTETomek vs 리샘플 없이 `scale_pos_weight`(neg/pos).

```bash
python scripts/compare_preprocessing_strategies.py
```

전체 CSV(약 28만 행)는 SMOTETomek 때문에 **수 분~수십 분** 걸릴 수 있다. 먼저 빠르게 돌려 보려면 앞 N행만 쓴다.

```bash
python scripts/compare_preprocessing_strategies.py --max-rows 50000
```

보고서용 최종 수치는 `--max-rows` 없이 전체 데이터로 한 번 돌리는 편이 안전하다.

설명은 `docs/leakage_and_scaling.md`와 맞춰 두었다. `fds_preprocessing_walkthrough.ipynb` 부록과도 같은 맥락이다.

---

## 지표

사기 거래는 전체에서 극소수라서, **맞춘 비율(Accuracy)만** 보면 “거의 다 정상으로만 맞춰도 점수가 높게” 나온다. 그래서 이 프로젝트에서는 주로 아래를 본다.

- **Recall**: 실제 사기 중 모델이 사기라고 잡아낸 비율. 놓치면 안 되는 경우에 중요하다.  
- **F1**: Precision(헛탐을 줄이는 쪽)과 Recall을 **같은 비중(β=1)**으로 섞은 값이다.  
- **F2**: Recall에 더 무게를 둔(β=2) 값이다. 미탐 비용을 더 크게 볼 때 쓰기 좋다.  
- **PR-AUC**: 임계값을 바꿔 가며 그린 정밀도–재현율 곡선의 면적에 가깝다. **순위가 얼마나 나은지**를 한 번에 보는 참고 지표로 쓴다. 불균형 데이터에서는 ROC-AUC만큼 낙관적이지 않은 경우가 많다.

Streamlit 사이드바와 `model_comparison.py`의 튜닝 목표는 **F2**로 맞춰 두었다. 대시보드에서 임계값 슬라이더를 움직이면 Recall·F1·F2는 그에 맞게 바뀌고, PR-AUC는 모델 순위 자체를 나타내는 값이라 임계값과 무관하게 한 번 찍힌 값으로 둔다.

---

## 동작 요약

- **전처리** (`fds_pipeline.py`): 원시 `Time`은 하루 주기로 보고 sin·cos 두 열로 바꾼 뒤 `Time` 열은 뺀다. `Amount`는 이상치에 덜 민감한 `RobustScaler`로 맞춘다. 스케일러는 전체 데이터프레임에 맞춘 뒤, 그다음 분할에 쓴다.  
- **분할**: 층화 추출로 Train 80% / Test 20%, 난수 시드는 42로 고정한다. SMOTETomek은 Train에만 적용한다.  
- **모델**: `fds_pipeline.train_xgb_pipeline` 안에서 고정된 하이퍼파라미터의 XGBoost를 학습한다.  
- **대시보드** (`app.py`): 임계값을 바꿀 때마다 Recall·F1·F2를 다시 계산한다. SHAP은 트리 전용 Explainer 대신 **예측 확률을 주는 함수**에 대한 Explainer를 쓰고, 그림은 `shap_waterfall_style.py`에서 글꼴·레이아웃을 다듬는다.

---

## 디렉터리

| 경로 | 역할 |
|------|------|
| `app.py` | Streamlit UI, 지표·SHAP 표시 |
| `fds_pipeline.py` | CSV 읽기, 전처리, 분할, 리샘플, XGB 학습 |
| `shap_waterfall_style.py` | SHAP 워터폴 그림을 화면에 맞게 보정 |
| `scripts/train_save_artifacts.py` | 학습 결과를 pkl로 저장 |
| `scripts/model_comparison.py` | 여러 모델 hold-out 비교 |
| `scripts/model_comparison_visualize.py` | 5모델 비교(ROC/PR/막대/혼동/보정/임계값) |
| `scripts/model_comparison_round2.py` | RF·XGB 3종, 검증 F2 최대 임계값 후 test 평가 |
| `scripts/compare_preprocessing_strategies.py` | 스케일·불균형 전략 비교(동일 split) |
| `report/export_report_bundle.py` | 보고서용 CSV·MD·JSON 묶음 생성 |
| `report/generate_report_*.py` | 보고서 삽입용 PNG 생성 |
| `docs/` | 보고서 문장·용어·누출/스케일 메모 |

---

## 라이선스·데이터 사용

데이터는 Kaggle이 제공한다. 다운로드·인용·재배포 규칙은 [데이터셋 페이지](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)와 Kaggle 약관을 보면 된다. 이 레포에는 원본 CSV를 넣지 않는다.

코드는 이 저장소에 `LICENSE` 파일을 두지 않았다. 학습·포트폴리오 참고용으로 올려 둔 것이고, 재사용·공개할 때는 본인이 라이선스를 정하면 된다.
