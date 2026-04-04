# SHAP 표현 (코드와 맞추기)

## 구현

- `app.py`: `shap.Explainer(fraud_proba, background)` — `predict_proba[:,1]` callable, 배경은 `X_test` 최대 500건.
- XGB 2.x TreeExplainer 직렬화 이슈를 피하려고 확률 기반 Explainer를 쓴다. “TreeExplainer만 썼다”고 쓰면 코드와 어긋날 수 있다.

## 보고서 예시 문장

> 단건 설명에 SHAP Explainer를 썼다. 사기 확률 함수와 배경 샘플을 넣어 워터폴로 기여를 그렸다.

## 피할 표현

- TreeExplainer만 사용 (단정)

## 후처리

- `shap_waterfall_style.py`는 글꼴·라벨 정리용이고 SHAP 값 알고리즘과는 별개다.
