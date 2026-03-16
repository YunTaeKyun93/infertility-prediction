import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def preprocess_xgb(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
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
    print(f"✅ XGBoost 전처리 완료: {df.shape}")
    return df, target, encoders


# ════════════════════════════════════════════════════════
# 4. Optuna 튜닝 (기존과 동일 — 학습 데이터 전체로 3-Fold CV)
# ════════════════════════════════════════════════════════
def tune_xgb(X, y, n_trials: int = 50):
    print(f"\n🔬 XGBoost Optuna 튜닝 ({n_trials} trials)...")
    pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "verbosity": 0, "scale_pos_weight": pos_weight,
            "n_estimators": 1000, "random_state": 42,
            "early_stopping_rounds": 50,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                      verbose=False)
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], pred))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  ➜ XGB 최적 AUC: {study.best_value:.4f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


def tune_cat(X, y, cat_cols, n_trials: int = 50):
    print(f"\n🔬 CatBoost Optuna 튜닝 ({n_trials} trials)...")
    pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "loss_function": "Logloss", "eval_metric": "AUC",
            "random_seed": 42, "verbose": False,
            "scale_pos_weight": pos_weight,
            "early_stopping_rounds": 50,
            "iterations":          trial.suggest_int("iterations", 300, 1000),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":               trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength":     trial.suggest_float("random_strength", 0, 2),
            "border_count":        trial.suggest_int("border_count", 32, 255),
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
            val_pool   = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_cols)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool)
            pred = model.predict_proba(val_pool)[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], pred))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  ➜ CAT 최적 AUC: {study.best_value:.4f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


# ════════════════════════════════════════════════════════
# 5. ★ K-Fold 학습 (기존 단순 train/val 분할 → 5-Fold로 교체)
# ════════════════════════════════════════════════════════
def kfold_train(X_cat, X_xgb, y, X_cat_test, X_xgb_test,
                cat_cols, xgb_best, cat_best, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_weight = (y == 0).sum() / (y == 1).sum()

    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    test_xgb = np.zeros(len(X_xgb_test))
    test_cat = np.zeros(len(X_cat_test))

    # Optuna 최적 파라미터 적용
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "scale_pos_weight": pos_weight, "n_estimators": 1000,
        "random_state": 42, "verbosity": 0, "early_stopping_rounds": 50,
        **(xgb_best or {"learning_rate": 0.05, "max_depth": 6,
                        "subsample": 0.8, "colsample_bytree": 0.8}),
    }
    cat_params = {
        "loss_function": "Logloss", "eval_metric": "AUC",
        "random_seed": 42, "scale_pos_weight": pos_weight,
        "early_stopping_rounds": 50, "verbose": False,
        **(cat_best or {"learning_rate": 0.05, "depth": 6, "iterations": 1000}),
    }

    fold_aucs_xgb, fold_aucs_cat = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_xgb, y)):
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1} / {n_splits}")
        print(f"{'='*50}")

        # ── XGBoost ─────────────────────────────────
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_xgb.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_xgb.iloc[val_idx])[:, 1]
        test_xgb += xgb_model.predict_proba(X_xgb_test)[:, 1] / n_splits
        auc_xgb = roc_auc_score(y.iloc[val_idx], oof_xgb[val_idx])
        fold_aucs_xgb.append(auc_xgb)
        print(f"  ⚡ XGB  AUC: {auc_xgb:.4f}")

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

    print(f"\n{'='*50}")
    print(f"  XGB OOF AUC: {roc_auc_score(y, oof_xgb):.4f}  "
          f"(Fold 평균: {np.mean(fold_aucs_xgb):.4f})")
    print(f"  CAT OOF AUC: {roc_auc_score(y, oof_cat):.4f}  "
          f"(Fold 평균: {np.mean(fold_aucs_cat):.4f})")

    return oof_xgb, oof_cat, test_xgb, test_cat, cat_model


# ════════════════════════════════════════════════════════
# 6. 최적 가중치 탐색
# ════════════════════════════════════════════════════════
def find_best_weights(oof_xgb, oof_cat, y, step=0.05):
    print("\n🔎 최적 가중치 탐색 중...")
    best_auc, best_w = 0, None
    for w in np.arange(0, 1 + step, step):
        blended = oof_xgb * w + oof_cat * (1 - w)
        auc = roc_auc_score(y, blended)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"  최적 AUC: {best_auc:.4f} | XGB: {best_w:.2f} / CAT: {1-best_w:.2f}")
    return best_w


# ════════════════════════════════════════════════════════
# 7. SHAP 분석
# ════════════════════════════════════════════════════════
def analyze_shap(model, X_val, cat_cols, save_dir="."):
    print("\n🔍 SHAP 분석 중 (CatBoost 기준)...")
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

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

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
    N_TRIALS   = 50                 # Optuna 시도 횟수
    N_SPLITS   = 5                  # K-Fold 수

    # 1. 로드
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # 2. 전처리
    X_cat, y, cat_cols      = preprocess_cat(train_df, is_train=True)
    X_xgb, _, xgb_encoders  = preprocess_xgb(train_df, is_train=True)
    X_cat_test, _, _        = preprocess_cat(test_df, is_train=False)
    X_xgb_test, _, _        = preprocess_xgb(test_df, is_train=False, encoders=xgb_encoders)

    print(f"\n학습: {X_cat.shape} | 테스트: {X_cat_test.shape}")
    print(f"성공률: {y.mean():.3f}")

    # 3. Optuna 튜닝 (전체 학습 데이터 기준)
    xgb_best = tune_xgb(X_xgb, y, n_trials=N_TRIALS)
    cat_best  = tune_cat(X_cat, y, cat_cols, n_trials=N_TRIALS)

    # 4. ★ K-Fold 학습 (기존 단순 분할 → 5-Fold로 교체)
    oof_xgb, oof_cat, test_xgb, test_cat, last_cat_model = kfold_train(
        X_cat, X_xgb, y,
        X_cat_test, X_xgb_test,
        cat_cols, xgb_best, cat_best,
        n_splits=N_SPLITS
    )

    # 5. 최적 가중치 탐색
    best_w = find_best_weights(oof_xgb, oof_cat, y)

    # 6. 최종 예측
    final_pred = test_xgb * best_w + test_cat * (1 - best_w)
    oof_final  = oof_xgb * best_w + oof_cat * (1 - best_w)
    print(f"\n🏆 최종 OOF AUC: {roc_auc_score(y, oof_final):.4f}")

    # 7. SHAP 분석 (마지막 Fold CAT 기준)
    analyze_shap(last_cat_model, X_cat_test, cat_cols, save_dir=SAVE_DIR)

    # 8. 제출 파일
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_kfold_optuna.csv", index=False)
    print("✅ submission_kfold_optuna.csv 저장 완료")
