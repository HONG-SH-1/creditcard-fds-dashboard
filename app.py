from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import shap
import streamlit as st
import time
from sklearn.metrics import average_precision_score, fbeta_score, recall_score

from fds_pipeline import CSV_DEFAULT, train_xgb_pipeline
from shap_waterfall_style import plot_waterfall_row

# pkl 있으면 로드, 없으면 학습
ARTIFACT_DIR = Path("artifacts")
ARTIFACTS = {
    "model": ARTIFACT_DIR / "fds_model.pkl",
    "scaler": ARTIFACT_DIR / "fds_scaler.pkl",
    "X_test": ARTIFACT_DIR / "fds_X_test.pkl",
    "y_test": ARTIFACT_DIR / "fds_y_test.pkl",
}


def _artifacts_ready() -> bool:
    return all(p.is_file() for p in ARTIFACTS.values())


def _shap_explainer(model, X_test):
    # XGB 2.x TreeExplainer 직렬화 이슈 회피 → predict_proba 기반 Explainer
    bg = X_test.iloc[: min(500, len(X_test))]

    def fraud_proba(X):
        return model.predict_proba(X)[:, 1]

    return shap.Explainer(fraud_proba, bg)


def _train_from_scratch():
    model, X_test, y_test, scaler = train_xgb_pipeline(CSV_DEFAULT)
    y_prob = model.predict_proba(X_test)[:, 1]
    explainer = _shap_explainer(model, X_test)
    return model, X_test, y_test, scaler, y_prob, explainer


def _load_from_artifacts():
    model = joblib.load(ARTIFACTS["model"])
    scaler = joblib.load(ARTIFACTS["scaler"])
    X_test = joblib.load(ARTIFACTS["X_test"])
    y_test = joblib.load(ARTIFACTS["y_test"])
    y_prob = model.predict_proba(X_test)[:, 1]
    explainer = _shap_explainer(model, X_test)
    return model, X_test, y_test, scaler, y_prob, explainer


@st.cache_resource
def load_model_bundle():
    if _artifacts_ready():
        return _load_from_artifacts()
    return _train_from_scratch()


st.set_page_config(page_title="AI-FDS Dashboard", layout="wide")
st.title("AI-FDS 실시간 모니터링 대시보드")

msg = (
    "저장된 모델(pkl) 로딩 중..."
    if _artifacts_ready()
    else "학습 데이터로 전체 학습 중... (시연 전: python scripts/train_save_artifacts.py)"
)
with st.spinner(msg):
    model, X_test, y_test, scaler, y_prob_all, explainer = load_model_bundle()

st.sidebar.header("리스크 컨트롤 패널")
threshold = st.sidebar.slider(
    "사기 판별 임계값 (Threshold)",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="낮추면 Recall↑, 오탐(FP)도 늘 수 있음.",
)

y_pred_dynamic = (y_prob_all >= threshold).astype(int)
current_recall = recall_score(y_test, y_pred_dynamic)
current_f2 = fbeta_score(y_test, y_pred_dynamic, beta=2)
pr_auc = average_precision_score(y_test, y_prob_all)

st.sidebar.markdown("---")
st.sidebar.subheader(f"현 임계값({threshold}) 기준 성능")
st.sidebar.info(
    f"Recall: {current_recall:.4f}\n\nF2: {current_f2:.4f}\n\nPR-AUC(고정): {pr_auc:.4f}"
)

st.sidebar.markdown("---")
st.sidebar.subheader("트랜잭션 단건 검증 (Demo)")
test_type = st.sidebar.radio("불러올 트랜잭션 유형", ["사기 의심 거래 (Fraud)", "정상 거래 (Normal)"])

if st.sidebar.button("랜덤 트랜잭션 불러오기"):
    target_class = 1 if test_type == "사기 의심 거래 (Fraud)" else 0
    sample_idx = y_test[y_test == target_class].sample(1, random_state=int(time.time())).index[0]
    sample_data = X_test.loc[[sample_idx]]

    fraud_prob = model.predict_proba(sample_data)[0][1]
    is_fraud = fraud_prob >= threshold

    if is_fraud:
        st.error(f"[FDS 차단 권고] 사기 의심 거래입니다! (위험도: {fraud_prob*100:.1f}%)")
    else:
        st.success(f"[정상 승인] 안전한 거래입니다. (위험도: {fraud_prob*100:.1f}%)")

    st.subheader("트랜잭션 요약")
    original_amount = scaler.inverse_transform(sample_data[["Amount"]])[0][0]
    sum_l, sum_r = st.columns(2)
    with sum_l:
        st.metric(label="결제 요청 금액", value=f"${original_amount:,.2f}")
    with sum_r:
        st.metric(
            label="AI 산출 사기 확률",
            value=f"{fraud_prob*100:.2f}%",
            delta=f"{-(threshold - fraud_prob)*100:.1f}% from Threshold",
            delta_color="inverse",
        )

    st.markdown("---")
    st.subheader("AI 판별 근거 (SHAP 분석)")
    st.caption(
        "변수별 기여(워터폴). 하단 기준은 기대 출력, f(x)는 이 건 사기 확률, 뒤 숫자는 해당 지점 값."
    )
    fig = plot_waterfall_row(explainer, sample_data, fraud_prob)
    st.pyplot(fig, use_container_width=True)
    plt.clf()
