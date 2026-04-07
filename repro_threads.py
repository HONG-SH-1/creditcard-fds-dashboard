# numpy/sklearn/xgboost 로드 전에 import 하세요. BLAS/OpenMP 병렬로 인한 비결정적 부동소수점을 줄입니다.
import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")
