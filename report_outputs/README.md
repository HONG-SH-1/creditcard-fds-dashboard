# report_outputs/

`python report/export_report_bundle.py`를 한 번 실행하면 이 폴더에 아래 파일이 생긴다.

- `holdout_benchmark.csv` / `holdout_benchmark.md` — 같은 내용을 표(CSV)와 마크다운으로 둔 것. 임계값 0.5일 때의 Recall, F1, F2, PR-AUC 등이 들어간다.  
- `benchmark_meta.json` — 어떤 CSV를 썼는지, 난수 시드, 튜닝에서 나온 요약 같은 **메타데이터**다.  
- `RUN_INFO.txt` — 스크립트를 실행한 시각(로컬 기준).

실행하려면 프로젝트 루트에 `dataset/creditcard.csv`가 있어야 한다.

이 폴더를 Git에 올릴지는 본인 선택이다. 기본 설정은 `.gitignore`로 생성물을 제외해 두었다.
