# F1 vs F2 (보고서용 메모)

수치는 `report_outputs/` 실측으로 채운다.

## 정의

- **F1**: Precision·Recall 동일 가중(β=1). FP·FN을 대칭으로 볼 때 요약용.
- **F2**: Recall 가중이 더 큼. 미탐 비용을 더 볼 때.
- **PR-AUC**: 임계값 고정 없이 순위 품질. 불균형에서 ROC-AUC보다 덜 낙관적인 경우가 많다.

## 이 프로젝트

- 대시보드: Recall + F2, PR-AUC는 임계값과 무관이라 고정 표시.
- `scripts/model_comparison.py` 튜닝 CV 스코어: F2.
- F1은 같은 Test에서 참고로 같이 둔다.

## 실측 문장 예시

> 임계값 0.5 기준 F1·F2·Recall·PR-AUC는 `holdout_benchmark.csv`와 같다.
