# 저랑 승희님

import pandas as pd

# 데이터 읽기 (경로를 정확히 입력하세요)
df = pd.read_csv('train.csv')

# 1. 상위 5개 행 보기 (데이터가 어떻게 생겼는지)
print(df.head())

# 2. 컬럼별 정보 확인 (데이터 타입, 결측치 여부)
print(df.info())

# 3. 통계적 정보 확인 (평균, 최대, 최소, 분위수 등)
print(df.describe())


import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ────────────────────────────────────────────────
plt.rcParams["font.family"] = "AppleGothic"   # Mac
# plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════
def load_data(train_path: str, test_path: str = None):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path) if test_path else None
    print(f"✅ 학습 데이터: {train.shape}")
    if test is not None:
        print(f"✅ 테스트 데이터: {test.shape}")
    return train, test


# ════════════════════════════════════════════════════════
# 2. 전처리
# ════════════════════════════════════════════════════════
# 결측률이 너무 높아 제거할 컬럼 (95% 이상)
DROP_COLS = [
    "착상 전 유전 검사 사용 여부",  # 99% 결측
    "PGD 시술 여부",               # 99% 결측
    "PGS 시술 여부",               # 99% 결측
    "난자 해동 경과일",             # 99% 결측
]

# 나이 순서 매핑
AGE_ORDER = {
    "만18-34세": 0,
    "만35-37세": 1,
    "만38-39세": 2,
    "만40-42세": 3,
    "만43-44세": 4,
    "만45-50세": 5,
}

# 횟수 컬럼 순서 매핑 (예: "1회" → 1)
COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
    "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
]

def parse_count(val):
    """'1회', '2회', '3회 이상' 등 → 숫자"""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if "이상" in val:
        return int(val[0]) + 1   # 보수적으로 +1
    digits = "".join(filter(str.isdigit, val))
    return int(digits) if digits else np.nan


def preprocess(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
    df = df.copy()

    # ── 타깃 분리 ──────────────────────────────────────
    target = None
    if is_train and "임신 성공 여부" in df.columns:
        target = df.pop("임신 성공 여부")

    # ── ID 제거 ────────────────────────────────────────
    df.drop(columns=["ID"], errors="ignore", inplace=True)

    # ── 고결측 컬럼 제거 ───────────────────────────────
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # ── 나이 Ordinal 인코딩 ────────────────────────────
    df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

    # ── 횟수 컬럼 수치화 ───────────────────────────────
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_count)

    # ── 결측 여부 이진 피처 추가 ───────────────────────
    # (결측 자체가 의미 있는 컬럼)
    missing_flag_cols = ["임신 시도 또는 마지막 임신 경과 연수"]
    for col in missing_flag_cols:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isna().astype(int)

    # ── 범주형 Label Encoding ──────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    # ── 수치형 결측값 중앙값 대체 ─────────────────────
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if is_train:
        medians = df[num_cols].median()
        encoders["medians"] = medians
    else:
        medians = encoders.get("medians", df[num_cols].median())

    df[num_cols] = df[num_cols].fillna(medians)

    print(f"✅ 전처리 완료: {df.shape} | 결측값: {df.isna().sum().sum()}")
    return df, target, encoders


# ════════════════════════════════════════════════════════
# 3. LightGBM 학습
# ════════════════════════════════════════════════════════
def train_lgb(X_train, y_train, X_val, y_val):
    params = {
        "objective":        "binary",
        "metric":           "auc",
        "learning_rate":    0.05,
        "num_leaves":       127,
        "max_depth":        -1,
        "min_child_samples": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),  # 불균형 보정
        "n_estimators":     1000,
        "random_state":     42,
        "verbose":          -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"\n✅ Validation AUC: {auc:.4f}")
    print(classification_report(y_val, (val_pred > 0.5).astype(int), target_names=["실패", "성공"]))
    return model


# ════════════════════════════════════════════════════════
# 4. SHAP 분석
# ════════════════════════════════════════════════════════
def analyze_shap(model, X_val: pd.DataFrame, save_dir: str = "."):
    print("\n🔍 SHAP 분석 시작...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # LightGBM binary: shap_values가 list면 [1] 인덱스 (성공 클래스)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ── ① Summary Plot (전체 피처 중요도) ─────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, show=False, max_display=20)
    plt.title("SHAP Summary Plot — 피처별 임신 성공 영향도", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Summary Plot 저장: {save_dir}/shap_summary.png")

    # ── ② Bar Plot (평균 절댓값 중요도) ───────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (평균 |SHAP|)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Importance Plot 저장: {save_dir}/shap_importance.png")

    # ── ③ 상위 피처 텍스트 출력 ───────────────────────
    mean_shap = pd.Series(
        np.abs(sv).mean(axis=0),
        index=X_val.columns
    ).sort_values(ascending=False)

    print("\n📋 상위 10개 중요 피처:")
    print(mean_shap.head(10).to_string())

    return sv, mean_shap


# ════════════════════════════════════════════════════════
# 5. 개별 케이스 분석 (선택)
# ════════════════════════════════════════════════════════
def analyze_single_case(model, X_val: pd.DataFrame, idx: int = 0, save_dir: str = "."):
    """특정 환자 1명의 예측 이유 분석"""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_val.iloc[[idx]])

    plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    plt.title(f"케이스 #{idx} 예측 분석", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_case_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 케이스 분석 저장: {save_dir}/shap_case_{idx}.png")


# ════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    TRAIN_PATH = "/Users/admin/nanim/infertility-prediction/train.csv"   # ← 경로 수정
    TEST_PATH  = "/Users/admin/nanim/infertility-prediction/test.csv"    # ← 없으면 None

    # 1. 로드
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # 2. 전처리
    X, y, encoders = preprocess(train_df, is_train=True)

    # 3. 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n학습: {X_train.shape}, 검증: {X_val.shape}")
    print(f"성공률 — 학습: {y_train.mean():.3f}, 검증: {y_val.mean():.3f}")

    # 4. LightGBM 학습
    model = train_lgb(X_train, y_train, X_val, y_val)

    # 5. SHAP 분석
    shap_vals, importance = analyze_shap(model, X_val, save_dir=".")

    # 6. 개별 케이스 (첫 번째 샘플)
    analyze_single_case(model, X_val, idx=0, save_dir=".")

    # 7. 테스트셋 예측 (있을 경우)
    if test_df is not None:
        X_test, _, _ = preprocess(test_df, is_train=False, encoders=encoders)
        test_pred = model.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame({
            "ID": test_df["ID"],
            "임신 성공 여부": test_pred
        })
        submission.to_csv("submission.csv", index=False)
        print("\n✅ submission.csv 저장 완료")
