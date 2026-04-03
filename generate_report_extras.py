"""
보고서 그림 7·8용 PNG (노트북과 동일 파일명).
- 07_top_v_kde_by_class.png : Class와 상관 큰 V 상위 3개, 정상 vs 사기 KDE (원시 스케일 EDA)
- 08_smotetomek_pca_before_after.png : fds_pipeline 전처리·분할 후 Train 리샘플 전·후 PCA 2D
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from fds_pipeline import (
    CSV_DEFAULT,
    RANDOM_STATE,
    preprocess_creditcard_dataframe,
    resample_train_smotetomek,
    stratified_train_test_split_creditcard,
)
from report_figures_common import REPORT_FIGURES_OUT, setup_korean_matplotlib_font

CSV_PATH = CSV_DEFAULT
OUT = REPORT_FIGURES_OUT


def fig_top_v_kde(df_raw: pd.DataFrame) -> None:
    v_cols = [c for c in df_raw.columns if c.startswith("V")]
    if not v_cols:
        raise ValueError("V 컬럼 없음")
    corr = df_raw[v_cols + ["Class"]].corr(numeric_only=True)["Class"].drop("Class").abs()
    top3 = corr.sort_values(ascending=False).head(3).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col in zip(axes, top3):
        for label, name, color in [(0, "정상(0)", "#4a90a4"), (1, "사기(1)", "#c44e4e")]:
            subset = df_raw.loc[df_raw["Class"] == label, col]
            sns.kdeplot(subset, ax=ax, label=name, color=color, fill=True, alpha=0.25, warn_singular=False)
        ax.set_title(f"{col} (|corr(Class)|={corr[col]:.3f})")
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
    fig.suptitle("상관 상위 V변수: 정상 vs 사기 분포 (KDE)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "07_top_v_kde_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", OUT / "07_top_v_kde_by_class.png")


def fig_smotetomek_pca(df_raw: pd.DataFrame) -> None:
    X, y, _ = preprocess_creditcard_dataframe(df_raw)
    X_train, _, y_train, _ = stratified_train_test_split_creditcard(X, y)
    # 전체 Train에 SMOTETomek는 수분 이상 걸릴 수 있어, 동일 분포의 층화 표본으로 패턴만 시각화
    cap = min(60_000, len(X_train))
    if len(X_train) > cap:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=cap, stratify=y_train, random_state=RANDOM_STATE
        )
    X_res, y_res = resample_train_smotetomek(X_train, y_train)
    note = f" (시각화용 Train 표본 n={len(y_train):,})"

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_before = pca.fit_transform(X_train)
    Z_after = pca.transform(X_res)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, Z, yy, title in [
        (axes[0], Z_before, y_train, "리샘플링 전 (Train)"),
        (axes[1], Z_after, y_res, "SMOTETomek 후 (Train)"),
    ]:
        m = yy == 0
        f = yy == 1
        ax.scatter(Z[m, 0], Z[m, 1], c="#4a90a4", s=3, alpha=0.35, label="정상(0)")
        ax.scatter(Z[f, 0], Z[f, 1], c="#c44e4e", s=8, alpha=0.65, label="사기(1)")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2)
    var = pca.explained_variance_ratio_
    fig.suptitle(
        f"SMOTETomek 전·후 Train PCA 투영 — 분산 {var[0]:.1%}, {var[1]:.1%}{note}",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT / "08_smotetomek_pca_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장:", OUT / "08_smotetomek_pca_before_after.png")


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"{CSV_PATH} 가 없습니다.")
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    setup_korean_matplotlib_font()

    df_raw = pd.read_csv(CSV_PATH)
    fig_top_v_kde(df_raw)
    fig_smotetomek_pca(df_raw)
    print("완료.")


if __name__ == "__main__":
    main()
