# report_outputs/

`python report/export_report_bundle.py` 한 번 실행 시 생성.

- `holdout_benchmark.csv` / `.md` — 임계값 0.5 기준 지표
- `benchmark_meta.json` — seed, 경로, 튜닝 요약
- `RUN_INFO.txt` — 실행 시각

`dataset/creditcard.csv` 필요.

Git에 넣을지는 선택(`.gitignore`로 제외해 둠).
