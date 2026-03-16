"""
불임 예측 앙상블 파이프라인 v3
LightGBM + XGBoost + CatBoost
→ 피처 엔지니어링 강화 (1등 코드 분석 반영)
→ Optuna 튜닝 + 5-Fold K-Fold
→ Nelder-Mead 최적 가중치 앙상블
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ════════════════════════════════════════════════════════
# 설정값
# ════════════════════════════════════════════════════════
SEED       = 42
N_TRIALS   = 50    # Optuna 시도 횟수 (시간 여유 있으면 100)
TUNE_FOLDS = 3     # Optuna 내부 CV Fold 수
N_FOLDS    = 5     # 최종 학습 K-Fold 수


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
# 2. 전처리 + 피처 엔지니어링 (1등 코드 + 기존 코드 통합)
# ════════════════════════════════════════════════════════
HIGH_NULL_COLS = [
    "난자 해동 경과일", "PGS 시술 여부", "PGD 시술 여부",
    "착상 전 유전 검사 사용 여부", "임신 시도 또는 마지막 임신 경과 연수",
    "불임 원인 - 여성 요인", "불임 원인 - 정자 면역학적 요인",
]
AGE_MAP = {
    "만18-34세": 1, "만35-37세": 2, "만38-39세": 3,
    "만40-42세": 4, "만43-44세": 5, "만45-50세": 6, "알 수 없음": -1
}
DONOR_MAP = {
    "만20세 이하": 1, "만21-25세": 2, "만26-30세": 3,
    "만31-35세": 4,  "만36-40세": 5, "만41-45세": 6, "알 수 없음": -1
}
CNT_MAP = {
    "0회": 0, "1회": 1, "2회": 2, "3회": 3,
    "4회": 4, "5회": 5, "6회 이상": 6
}
CNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
    "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
]
MALE_COLS = [
    "불임 원인 - 남성 요인", "불임 원인 - 정자 농도",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
]
FEMALE_COLS = [
    "불임 원인 - 난관 질환", "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
]
EPS = 1e-6


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    TARGET = "임신 성공 여부"
    ID_COL = "ID"

    y        = train[TARGET].copy()
    train_df = train.drop(columns=[TARGET, ID_COL])
    test_df  = test.drop(columns=[ID_COL])

    # train + test 합쳐서 전처리 (인코딩 일관성 보장)
    df      = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train = len(train_df)

    # ── 고결측 컬럼 → 결측 여부 플래그로 변환 후 제거 ────
    for col in HIGH_NULL_COLS:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)
    df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns], inplace=True)

    # ── 경과일 결측 여부 플래그 (1등 코드 반영) ──────────
    for col in ["배아 이식 경과일", "난자 혼합 경과일", "난자 채취 경과일", "배아 해동 경과일"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)

    # ── 나이 Ordinal 인코딩 ───────────────────────────────
    df["시술 당시 나이_num"]  = df["시술 당시 나이"].map(AGE_MAP)
    df["난자 기증자 나이_num"] = df["난자 기증자 나이"].map(DONOR_MAP)
    df["정자 기증자 나이_num"] = df["정자 기증자 나이"].map(DONOR_MAP)

    # ── 시술 횟수 Ordinal 인코딩 ─────────────────────────
    for col in CNT_COLS:
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(CNT_MAP)

    # ── 파생 피처 (기존 + 1등 코드 통합) ─────────────────
    # 배아 효율
    df["배아_이식률"]   = df["이식된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["배아_저장률"]   = df["저장된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["수정_성공률"]   = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + EPS)
    df["난자_활용률"]   = df["혼합된 난자 수"]             / (df["수집된 신선 난자 수"] + EPS)
    df["ICSI_이식비율"] = df["미세주입 배아 이식 수"]       / (df["이식된 배아 수"] + EPS)
    df["전체_효율"]     = df["이식된 배아 수"]              / (df["수집된 신선 난자 수"] + EPS)
    df["배아_손실률"]   = 1 - (df["이식된 배아 수"] + df["저장된 배아 수"]) / (df["총 생성 배아 수"] + EPS)
    df["미세주입_배아_비율"] = df["미세주입에서 생성된 배아 수"] / (df["총 생성 배아 수"] + EPS)

    # 이식 타이밍
    df["배양_기간"]      = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    df["이식_빠름"]      = (df["배아 이식 경과일"] <= 3).astype(int)
    df["이식_Day5"]      = (df["배아 이식 경과일"] == 5).astype(int)

    # 나이 관련
    df["나이_x_배아수"]    = df["시술 당시 나이_num"] * df["이식된 배아 수"]
    df["나이_x_생성배아"]  = df["시술 당시 나이_num"] * df["총 생성 배아 수"]
    df["나이_x_이식경과일"] = df["시술 당시 나이_num"] * df["배아 이식 경과일"]
    df["나이_x_시술횟수"]  = df["시술 당시 나이_num"] * df["총 시술 횟수_num"]  # 1등 코드

    # 시술 경험 관련
    df["과거_임신율"]    = df["총 임신 횟수_num"]  / (df["총 시술 횟수_num"] + EPS)
    df["임신_출산율"]    = df["총 출산 횟수_num"]  / (df["총 임신 횟수_num"] + EPS)
    df["과거_성공_경험"] = (df["총 임신 횟수_num"] > 0).astype(int)
    df["클리닉_집중도"]  = df["클리닉 내 총 시술 횟수_num"] / (df["총 시술 횟수_num"] + EPS)  # 1등 코드

    # 불임 원인 집계 (1등 코드) ─────────────────────────
    m_cols = [c for c in MALE_COLS   if c in df.columns]
    f_cols = [c for c in FEMALE_COLS if c in df.columns]
    df["남성_불임_원인_수"] = df[m_cols].sum(axis=1)
    df["여성_불임_원인_수"] = df[f_cols].sum(axis=1)
    df["총_불임_원인_수"]   = df["남성_불임_원인_수"] + df["여성_불임_원인_수"]
    df["복합_불임_여부"]    = ((df["남성_불임_원인_수"] > 0) & (df["여성_불임_원인_수"] > 0)).astype(int)

    # 시술 유형 텍스트 파싱 (1등 코드) ──────────────────
    df["ICSI_포함"]       = df["특정 시술 유형"].str.contains("ICSI",       na=False).astype(int)
    df["BLASTOCYST_포함"] = df["특정 시술 유형"].str.contains("BLASTOCYST", na=False).astype(int)
    df["AH_포함"]         = df["특정 시술 유형"].str.contains("AH",         na=False).astype(int)
    df["FER_포함"]        = df["특정 시술 유형"].str.contains("FER",        na=False).astype(int)
    df["복합시술_여부"]   = df["특정 시술 유형"].str.contains("/",          na=False).astype(int)

    # 배아_이식률 구간화
    df["배아_이식률_구간"] = pd.cut(
        df["배아_이식률"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 99],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # ── 원본 문자열 컬럼 제거 ─────────────────────────────
    drop_str = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"] + CNT_COLS
    df.drop(columns=[c for c in drop_str if c in df.columns], inplace=True)

    # ── Label Encoding ────────────────────────────────────
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].fillna("missing").astype(str))

    # ── 수치형 결측 중앙값 대체 ───────────────────────────
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    X      = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)

    print(f"✅ 전처리 완료 | X: {X.shape} | X_test: {X_test.shape}")
    print(f"   scale_pos_weight: {(y==0).sum()/(y==1).sum():.4f}")
    return X, X_test, y


# ════════════════════════════════════════════════════════
# 3. Optuna 튜닝
# ════════════════════════════════════════════════════════
def tune_lgb(X, y, spw):
    print(f"\n🔬 LightGBM Optuna 튜닝 ({N_TRIALS} trials × {TUNE_FOLDS}-fold)...")

    def objective(trial):
        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "verbose": -1, "n_jobs": -1, "random_state": SEED,
            "scale_pos_weight": spw, "n_estimators": 2000,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 32, 256),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        skf  = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"  ➜ LGB 최적 AUC: {study.best_value:.5f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


def tune_xgb(X, y, spw):
    print(f"\n🔬 XGBoost Optuna 튜닝 ({N_TRIALS} trials × {TUNE_FOLDS}-fold)...")

    def objective(trial):
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist", "verbosity": 0,
            "n_jobs": -1, "random_state": SEED,
            "scale_pos_weight": spw, "n_estimators": 2000,
            "early_stopping_rounds": 100,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        }
        skf  = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            m = xgb.XGBClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  verbose=False)
            aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"  ➜ XGB 최적 AUC: {study.best_value:.5f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


def tune_cat(X, y, spw):
    print(f"\n🔬 CatBoost Optuna 튜닝 ({N_TRIALS} trials × {TUNE_FOLDS}-fold)...")

    def objective(trial):
        params = {
            "eval_metric": "Logloss", "od_type": "Iter", "od_wait": 50,
            "verbose": False, "random_seed": SEED, "thread_count": -1,
            "scale_pos_weight": spw, "iterations": 2000,
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":              trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":    trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count":       trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf":   trial.suggest_int("min_data_in_leaf", 1, 50),
        }
        skf  = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                  use_best_model=True)
            aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"  ➜ CAT 최적 AUC: {study.best_value:.5f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


# ════════════════════════════════════════════════════════
# 4. K-Fold 최종 학습
# ════════════════════════════════════════════════════════
def kfold_train(X, X_test, y, lgb_best, xgb_best, cat_best):
    spw = (y == 0).sum() / (y == 1).sum()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lgb_oof  = np.zeros(len(X))
    xgb_oof  = np.zeros(len(X))
    cat_oof  = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))
    xgb_test = np.zeros(len(X_test))
    cat_test = np.zeros(len(X_test))

    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "verbose": -1, "n_jobs": -1, "random_state": SEED,
        "scale_pos_weight": spw, "n_estimators": 3000,
        **(lgb_best or {"learning_rate": 0.05, "num_leaves": 127}),
    }
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "verbosity": 0,
        "n_jobs": -1, "random_state": SEED,
        "scale_pos_weight": spw, "n_estimators": 3000,
        "early_stopping_rounds": 100,
        **(xgb_best or {"learning_rate": 0.05, "max_depth": 6}),
    }
    cat_params = {
        "eval_metric": "Logloss", "od_type": "Iter", "od_wait": 100,
        "verbose": False, "random_seed": SEED, "thread_count": -1,
        "scale_pos_weight": spw, "iterations": 3000,
        **(cat_best or {"learning_rate": 0.05, "depth": 6}),
    }

    # ── LightGBM ─────────────────────────────────────────
    print(f"\n🌿 LightGBM {N_FOLDS}-Fold 학습...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        lgb_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        lgb_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], lgb_oof[val_idx]):.5f}")
    print(f"  ➜ LGB OOF AUC: {roc_auc_score(y, lgb_oof):.5f}")

    # ── XGBoost ──────────────────────────────────────────
    print(f"\n⚡ XGBoost {N_FOLDS}-Fold 학습...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              verbose=False)
        xgb_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        xgb_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], xgb_oof[val_idx]):.5f}")
    print(f"  ➜ XGB OOF AUC: {roc_auc_score(y, xgb_oof):.5f}")

    # ── CatBoost ─────────────────────────────────────────
    print(f"\n🐱 CatBoost {N_FOLDS}-Fold 학습...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        m = CatBoostClassifier(**cat_params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
              use_best_model=True)
        cat_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        cat_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], cat_oof[val_idx]):.5f}")
    print(f"  ➜ CAT OOF AUC: {roc_auc_score(y, cat_oof):.5f}")

    return lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test


# ════════════════════════════════════════════════════════
# 5. Nelder-Mead 최적 가중치 앙상블
# ════════════════════════════════════════════════════════
def optimize_weights(lgb_oof, xgb_oof, cat_oof, y):
    print("\n🔎 Nelder-Mead 최적 가중치 탐색 중...")

    def neg_auc(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-8)
        return -roc_auc_score(y, w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof)

    result = minimize(neg_auc, x0=[1/3, 1/3, 1/3],
                      method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-7})
    opt_w = np.clip(result.x, 0, None)
    opt_w = opt_w / opt_w.sum()

    lgb_auc   = roc_auc_score(y, lgb_oof)
    xgb_auc   = roc_auc_score(y, xgb_oof)
    cat_auc   = roc_auc_score(y, cat_oof)
    equal_auc = roc_auc_score(y, (lgb_oof + xgb_oof + cat_oof) / 3)
    opt_auc   = roc_auc_score(y, opt_w[0]*lgb_oof + opt_w[1]*xgb_oof + opt_w[2]*cat_oof)

    print("=" * 50)
    print("최종 OOF AUC 비교")
    print("=" * 50)
    print(f"  LightGBM 단독:    {lgb_auc:.5f}")
    print(f"  XGBoost  단독:    {xgb_auc:.5f}")
    print(f"  CatBoost 단독:    {cat_auc:.5f}")
    print(f"  균등 앙상블:      {equal_auc:.5f}")
    print(f"  최적가중치 앙상블: {opt_auc:.5f}  ← 최종 제출")
    print(f"  최적 가중치: LGB={opt_w[0]:.3f} / XGB={opt_w[1]:.3f} / CAT={opt_w[2]:.3f}")

    return opt_w, opt_auc


# ════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"   # ← 경로 수정
    TEST_PATH  = "data/test.csv"
    SAVE_DIR   = "."

    # 1. 로드
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # 2. 전처리
    X, X_test, y = preprocess(train_df, test_df)
    spw = (y == 0).sum() / (y == 1).sum()

    # 3. Optuna 튜닝
    lgb_best = tune_lgb(X, y, spw)
    xgb_best = tune_xgb(X, y, spw)
    cat_best = tune_cat(X, y, spw)

    # 4. K-Fold 최종 학습
    lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test = kfold_train(
        X, X_test, y, lgb_best, xgb_best, cat_best
    )

    # 5. 최적 가중치 앙상블
    opt_w, opt_auc = optimize_weights(lgb_oof, xgb_oof, cat_oof, y)

    # 6. 최종 예측 및 제출
    final_pred = opt_w[0]*lgb_test + opt_w[1]*xgb_test + opt_w[2]*cat_test
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_v3.csv", index=False)
    print(f"\n✅ submission_v3.csv 저장 완료  (OOF AUC: {opt_auc:.5f})")
