# 누출(leakage)·처리 순서·Amount 스케일 선택

이 문서는 `fds_pipeline.py`, `fds_preprocessing_walkthrough.ipynb`, `README.md`의 말을 **한곳에 맞춰** 두기 위한 것이다.  
수치 비교는 `python scripts/compare_preprocessing_strategies.py`로 같은 데이터·같은 분할에서 재현할 수 있다.

## 파이프라인 순서 (시간 순)

1. CSV 로드  
2. `Time` → `Time_sin`, `Time_cos` 후 원시 `Time` 제거  
3. `Amount`에 `RobustScaler` 적용  
4. `X`, `y` (`Class`) 분리  
5. **층화**로 Train / Test 분할 (`test_size=0.2`, `random_state=42`)  
6. **Train에만** SMOTETomek (Test는 원 불균형 유지)  
7. XGBoost 학습 → Test로 지표  

리샘플은 **분할 뒤·Train에만** 적용되므로, Test 분포가 학습용 합성 샘플에 섞이지 않는다.

## Amount 스케일: full df fit vs train-only fit

### 일반적으로 자주 권하는 것

Train에만 `fit`하고 Test는 `transform`만 하는 방식은 **테스트 라벨이 스케일 통계에 스며들지 않게** 하려는 전형적인 패턴이다.

### 이 프로젝트의 구현

`preprocess_creditcard_dataframe`에서는 **Train/Test 나누기 전**에 전체 행에 대해 `RobustScaler`를 `fit`한다.  
즉, **통계량(중앙값·IQR 등) 계산에 Test 구간의 `Amount` 값이 포함**된다. 엄밀히 말하면 이는 “Train만 본다”는 규칙과 다르다.

**왜 이렇게 두었는지(설계 관점 예시)**

- 극단 금액 스케일을 전체 분포 기준으로 맞추려는 실험적 선택  
- 과제·포폴 범위에서 **의도적으로 고정**해 두고, 대안(train-only)과 **수치로 비교**하는 것이 설명에 유리하다  

**권장**

- 보고서·면접에서는 위 차이를 **한 문단**으로 밝힌다.  
- `compare_preprocessing_strategies.py` 결과 표로 **full-fit vs train-only** Test 지표를 나란히 두면 “알고 선택했다”는 근거가 된다.

## 불균형: SMOTETomek vs `scale_pos_weight`

- **SMOTETomek**(현재 파이프라인): Train 소수 클래스를 합성·정리해 **표본 수 자체**를 바꾼 뒤 학습한다.  
- **`scale_pos_weight`**: 표본 수는 그대로 두고, XGBoost가 손실에서 클래스 비중을 조정한다(대략 `neg/pos` 비율).  

둘 다 불균형 대응이지만 메커니즘이 다르다. 스크립트 **실험 2**는 RobustScaler를 full-fit으로 **통일한 뒤** 두 전략만 바꿔 같은 Test 지표를 비교한다.

## 재현성

- 분할·SMOTE·XGB 시드: `fds_pipeline`의 `RANDOM_STATE` 등과 동일하게 맞춘다.  
- 스크립트는 `dataset/creditcard.csv`가 있어야 한다.

## 실행 시간

`compare_preprocessing_strategies.py`는 SMOTETomek을 여러 번 호출한다. **전체 행**이면 CPU·환경에 따라 **매우 오래** 걸릴 수 있어, 멈춘 것처럼 보일 수 있다.  
먼저 `python scripts/compare_preprocessing_strategies.py --max-rows 50000` 처럼 **앞 N행만**으로 동작을 확인하고, 보고서용으로는 시간을 내어 `--max-rows` 없이 전체를 돌리면 된다.
