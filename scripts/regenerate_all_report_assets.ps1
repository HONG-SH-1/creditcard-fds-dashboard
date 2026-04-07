# 보고서용 산출물 전부 재생성 (순서대로, 오래 걸릴 수 있음)
# 프로젝트 루트에서:  powershell -ExecutionPolicy Bypass -File scripts/regenerate_all_report_assets.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "1) 아티팩트 (artifacts/*.pkl)"
python scripts/train_save_artifacts.py

Step "2) EDA·보조 플롯 (report_figures/eda/)"
python scripts/report_plots/plot_eda_class_distribution.py
python scripts/report_plots/plot_eda_correlation_heatmap.py
python scripts/report_plots/plot_eda_time_kde.py
python scripts/report_plots/plot_eda_amount.py
python scripts/report_plots/plot_preprocess_time_sin_cos.py
python scripts/report_plots/plot_ch8_descriptive_statistics.py

Step "3) 1차 벤치마크 + 시각화 (report_figures 루트)"
python scripts/model_comparison.py
python scripts/model_comparison_visualize.py --out-dir report_figures

Step "4) 2차 + val F2 곡선"
python scripts/model_comparison_round2.py
python scripts/report_plots/plot_round2_val_f2_vs_threshold.py

Step "5) 도식·학습곡선"
python report/generate_report_model_diagrams.py

Step "6) (선택) extras"
python report/generate_report_extras.py

Step "7) (선택) 전처리 민감도"
python scripts/compare_preprocessing_strategies.py

Write-Host "`n끝. fds_report_figures.ipynb 는 Jupyter에서 수동 실행." -ForegroundColor Green
