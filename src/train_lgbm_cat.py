import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "AppleGothic"
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
# 2. 피처 엔지니어링
# ════════════════════════════════════════════════════════
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["배아_이식률"]    = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)
    df["배아_저장률"]    = df["저장된 배아 수"] / (df["총 생성 배아 수"] + 1)
    df["ICSI_이식비율"]  = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)
    df["수정_성공률"]    = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + 1)
    df["난자_활용률"]    = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)
    df["나이_x_배아수"]  = df["시술 당시 나이"] * df["이식된 배아 수"]
    df["나이_x_생성배아"] = df["시술 당시 나이"] * df["총 생성 배아 수"]
    df["과거_임신율"]    = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)
    df["임신_출산율"]    = df["총 출산 횟수"] / (df["총 임신 횟수"] + 1)
    df["배양_기간"]      = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    df["이식_빠름"]      = (df["배아 이식 경과일"] <= 3).astype(int)
    return df


# ════════════════════════════════════════════════════════
# 3. 전처리
# ════════════════════════════════════════════════════════
DROP_COLS = [
    "착상 전 유전 검사 사용 여부", "PGD 시술 여부",
    "PGS 시술 여부", "난자 해동 경과일",
    "불임 원인 - 여성 요인",
    "불임 원인 - 정자 면역학적 요인",
]
AGE_ORDER = {
    "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
    "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
}
COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
    "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
]

def parse_count(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if "이상" in val: return int(val[0]) + 1
    digits = "".join(filter(str.isdigit, val))
    return int(digits) if digits else np.nan

def _base_preprocess(df, is_train):
    target = None
    if is_train and "임신 성공 여부" in df.columns:
        target = df.pop("임신 성공 여부")
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)
    df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_count)
    for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isna().astype(int)
    df = feature_engineering(df)
    return df, target

def preprocess_cat(df: pd.DataFrame, is_train: bool = True):
    df = df.copy()
    df, target = _base_preprocess(df, is_train)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
    print(f"✅ CatBoost 전처리 완료: {df.shape}")
    return df, target, cat_cols

def preprocess_lgb(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
    df = df.copy()
    df, target = _base_preprocess(df, is_train)
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
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if is_train:
        medians = df[num_cols].median()
        encoders["medians"] = medians
    else:
        medians = encoders.get("medians", df[num_cols].median())
    df[num_cols] = df[num_cols].fillna(medians)
    print(f"✅ LightGBM 전처리 완료: {df.shape}")
    return df, target, encoders


# ════════════════════════════════════════════════════════
# 4. K-Fold 앙상블 학습
# ════════════════════════════════════════════════════════
def kfold_train(X_cat, X_lgb, y, X_cat_test, X_lgb_test,
                cat_cols, n_splits=5):
    """
    5-Fold K-Fold로 LGB + CAT 학습
    - 각 Fold마다 검증셋 예측값을 OOF(Out-Of-Fold)로 저장
    - 테스트셋 예측값은 5개 Fold 평균
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_weight = (y == 0).sum() / (y == 1).sum()

    # OOF 예측 저장 배열
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))

    # 테스트셋 예측 저장 배열 (Fold별 평균)
    test_lgb = np.zeros(len(X_lgb_test))
    test_cat = np.zeros(len(X_cat_test))

    lgb_params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 127,
        "min_child_samples": 50, "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "n_estimators": 1000, "random_state": 42, "verbose": -1,
    }
    cat_params = {
        "loss_function": "Logloss", "eval_metric": "AUC",
        "learning_rate": 0.05, "depth": 6, "iterations": 1000,
        "random_seed": 42, "scale_pos_weight": pos_weight,
        "early_stopping_rounds": 50, "verbose": False,
    }

    fold_aucs_lgb, fold_aucs_cat = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb, y)):
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1} / {n_splits}")
        print(f"{'='*50}")

        # ── LightGBM ────────────────────────────────
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_lgb.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_lgb.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(200)],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_lgb.iloc[val_idx])[:, 1]
        test_lgb += lgb_model.predict_proba(X_lgb_test)[:, 1] / n_splits
        auc_lgb = roc_auc_score(y.iloc[val_idx], oof_lgb[val_idx])
        fold_aucs_lgb.append(auc_lgb)
        print(f"  🌿 LGB  AUC: {auc_lgb:.4f}")

        # ── CatBoost ────────────────────────────────
        train_pool = Pool(X_cat.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        val_pool   = Pool(X_cat.iloc[val_idx], y.iloc[val_idx], cat_features=cat_cols)
        test_pool  = Pool(X_cat_test, cat_features=cat_cols)

        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(train_pool, eval_set=val_pool)
        oof_cat[val_idx] = cat_model.predict_proba(val_pool)[:, 1]
        test_cat += cat_model.predict_proba(test_pool)[:, 1] / n_splits
        auc_cat = roc_auc_score(y.iloc[val_idx], oof_cat[val_idx])
        fold_aucs_cat.append(auc_cat)
        print(f"  🐱 CAT  AUC: {auc_cat:.4f}")

    # ── 전체 OOF AUC ────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  LGB  OOF AUC: {roc_auc_score(y, oof_lgb):.4f}  "
          f"(Fold 평균: {np.mean(fold_aucs_lgb):.4f})")
    print(f"  CAT  OOF AUC: {roc_auc_score(y, oof_cat):.4f}  "
          f"(Fold 평균: {np.mean(fold_aucs_cat):.4f})")

    return oof_lgb, oof_cat, test_lgb, test_cat


# ════════════════════════════════════════════════════════
# 5. 최적 가중치 탐색
# ════════════════════════════════════════════════════════
def find_best_weights(oof_lgb, oof_cat, y, step=0.05):
    print("\n🔎 최적 가중치 탐색 중...")
    best_auc, best_w = 0, None
    for w in np.arange(0, 1 + step, step):
        blended = oof_lgb * w + oof_cat * (1 - w)
        auc = roc_auc_score(y, blended)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"  최적 AUC: {best_auc:.4f} | LGB: {best_w:.2f} / CAT: {1-best_w:.2f}")
    return best_w


# ════════════════════════════════════════════════════════
# 6. SHAP 분석 (마지막 Fold의 CAT 기준)
# ════════════════════════════════════════════════════════
def analyze_shap(model, X_val, cat_cols, save_dir="."):
    print("\n🔍 SHAP 분석 중...")
    val_pool    = Pool(X_val, cat_features=cat_cols)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(val_pool)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, show=False, max_display=20)
    plt.title("SHAP Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  📊 shap_summary.png 저장")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  📊 shap_importance.png 저장")

    mean_shap = pd.Series(np.abs(sv).mean(axis=0),
                          index=X_val.columns).sort_values(ascending=False)
    print("\n📋 상위 15개 중요 피처:")
    print(mean_shap.head(15).to_string())


# ════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"   # ← 경로 수정
    TEST_PATH  = "data/test.csv"    # ← 없으면 None
    SAVE_DIR   = "."
    N_SPLITS   = 5                  # K-Fold 수

    # 1. 로드
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # 2. 전처리
    X_cat, y, cat_cols      = preprocess_cat(train_df, is_train=True)
    X_lgb, _, lgb_encoders  = preprocess_lgb(train_df, is_train=True)

    X_cat_test, _, _  = preprocess_cat(test_df, is_train=False)
    X_lgb_test, _, _  = preprocess_lgb(test_df, is_train=False, encoders=lgb_encoders)

    print(f"\n학습: {X_cat.shape} | 테스트: {X_cat_test.shape}")
    print(f"성공률: {y.mean():.3f}")

    # 3. K-Fold 학습
    oof_lgb, oof_cat, test_lgb, test_cat = kfold_train(
        X_cat, X_lgb, y,
        X_cat_test, X_lgb_test,
        cat_cols, n_splits=N_SPLITS
    )

    # 4. 최적 가중치 탐색
    best_w = find_best_weights(oof_lgb, oof_cat, y)

    # 5. 최종 예측
    final_pred = test_lgb * best_w + test_cat * (1 - best_w)
    oof_final  = oof_lgb * best_w + oof_cat * (1 - best_w)
    print(f"\n🏆 최종 OOF AUC: {roc_auc_score(y, oof_final):.4f}")

    # 6. 제출 파일
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_kfold.csv", index=False)
    print("✅ submission_kfold.csv 저장 완료")
