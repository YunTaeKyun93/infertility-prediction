"""
실패 케이스 집중 버전
① scale_pos_weight 조정
② 실패 전용 피처 추가
① Pseudo Labeling (0.95/0.05) 재학습
② 기존 제출 파일 4개 앙상블
③ ① + ② 최종 앙상블
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

SEED    = 42
N_FOLDS = 10

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "verbose": -1, "n_jobs": -1, "random_state": SEED,
    "n_estimators": 3000,
    "learning_rate": 0.05, "num_leaves": 127,
    "max_depth": 6, "min_child_samples": 50,
    "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "tree_method": "hist", "verbosity": 0,
    "n_jobs": -1, "random_state": SEED,
    "n_estimators": 3000, "early_stopping_rounds": 100,
    "learning_rate": 0.0639, "max_depth": 4,
    "subsample": 0.769, "colsample_bytree": 0.707,
    "colsample_bylevel": 0.8, "min_child_weight": 16,
    "gamma": 3.599, "reg_alpha": 9.720, "reg_lambda": 0.00169,
}
CAT_PARAMS = {
    "eval_metric": "Logloss", "od_type": "Iter", "od_wait": 100,
    "verbose": False, "random_seed": SEED, "thread_count": -1,
    "iterations": 3000,
    "learning_rate": 0.0612, "depth": 5,
    "l2_leaf_reg": 5.879, "bagging_temperature": 0.464,
    "random_strength": 0.731, "border_count": 117,
}

TARGET_ENCODE_COLS = [
    "배란 유도 유형", "특정 시술 유형", "난자 출처",
    "정자 출처", "시술 시기 코드", "배아 생성 주요 이유",
]
HIGH_NULL_COLS = [
    "난자 해동 경과일", "PGS 시술 여부", "PGD 시술 여부",
    "착상 전 유전 검사 사용 여부", "임신 시도 또는 마지막 임신 경과 연수",
    "불임 원인 - 여성 요인", "불임 원인 - 정자 면역학적 요인",
]
AGE_MAP    = {"만18-34세":1,"만35-37세":2,"만38-39세":3,"만40-42세":4,"만43-44세":5,"만45-50세":6,"알 수 없음":-1}
DONOR_MAP  = {"만20세 이하":1,"만21-25세":2,"만26-30세":3,"만31-35세":4,"만36-40세":5,"만41-45세":6,"알 수 없음":-1}
CNT_MAP    = {"0회":0,"1회":1,"2회":2,"3회":3,"4회":4,"5회":5,"6회 이상":6}
CNT_COLS   = ["총 시술 횟수","클리닉 내 총 시술 횟수","IVF 시술 횟수","DI 시술 횟수",
              "총 임신 횟수","IVF 임신 횟수","DI 임신 횟수","총 출산 횟수","IVF 출산 횟수","DI 출산 횟수"]
MALE_COLS  = ["불임 원인 - 남성 요인","불임 원인 - 정자 농도","불임 원인 - 정자 운동성","불임 원인 - 정자 형태"]
FEMALE_COLS= ["불임 원인 - 난관 질환","불임 원인 - 배란 장애","불임 원인 - 자궁경부 문제","불임 원인 - 자궁내막증"]
EPS = 1e-6


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"✅ 학습: {train.shape} | 테스트: {test.shape}")
    return train, test


def preprocess(train, test):
    TARGET, ID_COL = "임신 성공 여부", "ID"
    y        = train[TARGET].copy()
    train_df = train.drop(columns=[TARGET, ID_COL])
    test_df  = test.drop(columns=[ID_COL])
    df       = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train  = len(train_df)

    for col in HIGH_NULL_COLS:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)
    df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns], inplace=True)

    for col in ["배아 이식 경과일","난자 혼합 경과일","난자 채취 경과일","배아 해동 경과일"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)

    df["시술 당시 나이_num"]   = df["시술 당시 나이"].map(AGE_MAP)
    df["난자 기증자 나이_num"]  = df["난자 기증자 나이"].map(DONOR_MAP)
    df["정자 기증자 나이_num"]  = df["정자 기증자 나이"].map(DONOR_MAP)

    for col in CNT_COLS:
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(CNT_MAP)

    df["배아_이식률"]       = df["이식된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["배아_저장률"]       = df["저장된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["수정_성공률"]       = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + EPS)
    df["난자_활용률"]       = df["혼합된 난자 수"]             / (df["수집된 신선 난자 수"] + EPS)
    df["ICSI_이식비율"]     = df["미세주입 배아 이식 수"]       / (df["이식된 배아 수"] + EPS)
    df["전체_효율"]         = df["이식된 배아 수"]              / (df["수집된 신선 난자 수"] + EPS)
    df["배아_손실률"]       = 1 - (df["이식된 배아 수"] + df["저장된 배아 수"]) / (df["총 생성 배아 수"] + EPS)
    df["미세주입_배아_비율"] = df["미세주입에서 생성된 배아 수"] / (df["총 생성 배아 수"] + EPS)
    df["배양_기간"]         = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    df["이식_빠름"]         = (df["배아 이식 경과일"] <= 3).astype(int)
    df["이식_Day5"]         = (df["배아 이식 경과일"] == 5).astype(int)
    df["나이_x_배아수"]     = df["시술 당시 나이_num"] * df["이식된 배아 수"]
    df["나이_x_생성배아"]   = df["시술 당시 나이_num"] * df["총 생성 배아 수"]
    df["나이_x_이식경과일"] = df["시술 당시 나이_num"] * df["배아 이식 경과일"]
    df["나이_x_시술횟수"]   = df["시술 당시 나이_num"] * df["총 시술 횟수_num"]
    df["과거_임신율"]       = df["총 임신 횟수_num"]  / (df["총 시술 횟수_num"] + EPS)
    df["임신_출산율"]       = df["총 출산 횟수_num"]  / (df["총 임신 횟수_num"] + EPS)
    df["과거_성공_경험"]    = (df["총 임신 횟수_num"] > 0).astype(int)
    df["클리닉_집중도"]     = df["클리닉 내 총 시술 횟수_num"] / (df["총 시술 횟수_num"] + EPS)

    m_cols = [c for c in MALE_COLS   if c in df.columns]
    f_cols = [c for c in FEMALE_COLS if c in df.columns]
    df["남성_불임_원인_수"] = df[m_cols].sum(axis=1)
    df["여성_불임_원인_수"] = df[f_cols].sum(axis=1)
    df["총_불임_원인_수"]   = df["남성_불임_원인_수"] + df["여성_불임_원인_수"]
    df["복합_불임_여부"]    = ((df["남성_불임_원인_수"] > 0) & (df["여성_불임_원인_수"] > 0)).astype(int)

    df["ICSI_포함"]       = df["특정 시술 유형"].str.contains("ICSI",       na=False).astype(int)
    df["BLASTOCYST_포함"] = df["특정 시술 유형"].str.contains("BLASTOCYST", na=False).astype(int)
    df["AH_포함"]         = df["특정 시술 유형"].str.contains("AH",         na=False).astype(int)
    df["FER_포함"]        = df["특정 시술 유형"].str.contains("FER",        na=False).astype(int)
    df["복합시술_여부"]   = df["특정 시술 유형"].str.contains("/",          na=False).astype(int)

    df["배아_이식률_구간"] = pd.cut(
        df["배아_이식률"], bins=[-0.01,0.2,0.4,0.6,0.8,99], labels=[0,1,2,3,4]
    ).astype(float)

    # ── 실패 전용 피처 ────────────────────────────────
    # 고령 + 배아 적음 → 실패 위험 높음
    df["고령_저배아"] = ((df["시술 당시 나이_num"] >= 4) &
                        (df["이식된 배아 수"] <= 1)).astype(int)

    # 반복 실패 여부 (3회 이상 시술)
    df["반복실패_여부"] = (df["총 시술 횟수_num"] >= 3).astype(int)

    # 선택지 없이 다 이식 (배아_이식률 높음 = 선택지 없었음)
    df["저품질_배아"] = (df["배아_이식률"] >= 0.9).astype(int)

    # Day 1~2 이식 (너무 이른 이식 = 실패율 높음)
    df["이식_너무빠름"] = (df["배아 이식 경과일"] <= 2).astype(int)

    # 남성 + 여성 불임 원인 동시 존재 (복합 불임)
    df["복합_고령"] = ((df["복합_불임_여부"] == 1) &
                      (df["시술 당시 나이_num"] >= 3)).astype(int)

    # 총 생성 배아 적음 (난소 반응 약함)
    df["저반응_난소"] = (df["총 생성 배아 수"] <= 2).astype(int)

    print(f"  ✅ 실패 전용 피처 6개 추가")

    drop_str = ["시술 당시 나이","난자 기증자 나이","정자 기증자 나이"] + CNT_COLS
    df.drop(columns=[c for c in drop_str if c in df.columns], inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].fillna("missing").astype(str))

    for col in df.select_dtypes(include=["float64","int64"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    X      = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)
    spw    = (y==0).sum() / (y==1).sum()
    # ★ scale_pos_weight 조정 — 실패에 더 집중
    # 원래값(2.87)보다 낮게 → 실패 케이스를 더 잘 잡음
    spw = min(spw, 1.5)
    print(f"✅ 전처리 완료 | X: {X.shape} | X_test: {X_test.shape}")
    print(f"   scale_pos_weight: {spw:.4f} (조정됨)")
    return X, X_test, y, spw


def apply_target_encoding(X, X_test, y):
    X, X_test = X.copy(), X_test.copy()
    global_mean = y.mean()
    for col in TARGET_ENCODE_COLS:
        if col not in X.columns:
            continue
        mean_map = y.groupby(X[col]).mean()
        X[f"{col}_te"]      = X[col].map(mean_map).fillna(global_mean)
        X_test[f"{col}_te"] = X_test[col].map(mean_map).fillna(global_mean)
    print(f"✅ 타깃 인코딩 완료")
    return X, X_test


def select_features(X, X_test, y, spw, threshold_pct=10):
    print(f"\n🔍 피처 선택 중...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "scale_pos_weight":spw, "n_estimators":500})
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    importance = pd.Series(m.feature_importances_, index=X.columns)
    threshold  = importance.quantile(threshold_pct/100)
    keep_cols  = importance[importance > threshold].index.tolist()
    print(f"  {len(X.columns)}개 → {len(keep_cols)}개 (하위 {threshold_pct}% 제거)")
    return X[keep_cols], X_test[keep_cols], keep_cols


def kfold_train(X, X_test, y, spw, tag=""):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lgb_oof=np.zeros(len(X)); xgb_oof=np.zeros(len(X)); cat_oof=np.zeros(len(X))
    lgb_test=np.zeros(len(X_test)); xgb_test=np.zeros(len(X_test)); cat_test=np.zeros(len(X_test))

    lgb_p = {**LGB_PARAMS, "scale_pos_weight":spw}
    xgb_p = {**XGB_PARAMS, "scale_pos_weight":spw}
    cat_p = {**CAT_PARAMS, "scale_pos_weight":spw}

    print(f"\n🌿 LightGBM {N_FOLDS}-Fold {tag}...")
    for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
        m = lgb.LGBMClassifier(**lgb_p)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(500)])
        lgb_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:,1]
        lgb_test += m.predict_proba(X_test)[:,1] / N_FOLDS
        print(f"  Fold {fold+1:2d}: {roc_auc_score(y.iloc[val_idx], lgb_oof[val_idx]):.5f}")
    print(f"  ➜ LGB OOF: {roc_auc_score(y, lgb_oof):.5f}")

    print(f"\n⚡ XGBoost {N_FOLDS}-Fold {tag}...")
    for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
        m = xgb.XGBClassifier(**xgb_p)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
        xgb_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:,1]
        xgb_test += m.predict_proba(X_test)[:,1] / N_FOLDS
        print(f"  Fold {fold+1:2d}: {roc_auc_score(y.iloc[val_idx], xgb_oof[val_idx]):.5f}")
    print(f"  ➜ XGB OOF: {roc_auc_score(y, xgb_oof):.5f}")

    print(f"\n🐱 CatBoost {N_FOLDS}-Fold {tag}...")
    for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
        m = CatBoostClassifier(**cat_p)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]), use_best_model=True)
        cat_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:,1]
        cat_test += m.predict_proba(X_test)[:,1] / N_FOLDS
        print(f"  Fold {fold+1:2d}: {roc_auc_score(y.iloc[val_idx], cat_oof[val_idx]):.5f}")
    print(f"  ➜ CAT OOF: {roc_auc_score(y, cat_oof):.5f}")

    return lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test


def optimize_weights(lgb_oof, xgb_oof, cat_oof, y):
    def neg_auc(w):
        w = np.clip(w, 0, None); w = w/(w.sum()+1e-8)
        return -roc_auc_score(y, w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof)
    result = minimize(neg_auc, x0=[1/3,1/3,1/3], method="Nelder-Mead",
                      options={"maxiter":2000,"xatol":1e-7})
    opt_w = np.clip(result.x, 0, None); opt_w = opt_w/opt_w.sum()
    opt_auc = roc_auc_score(y, opt_w[0]*lgb_oof + opt_w[1]*xgb_oof + opt_w[2]*cat_oof)
    print(f"  최적가중치 앙상블: {opt_auc:.5f}")
    print(f"  가중치: LGB={opt_w[0]:.3f} / XGB={opt_w[1]:.3f} / CAT={opt_w[2]:.3f}")
    return opt_w, opt_auc


if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"
    TEST_PATH  = "data/test.csv"
    SAVE_DIR   = "."

    # 1. 로드 & 전처리
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    X, X_test, y, spw = preprocess(train_df, test_df)
    X, X_test = apply_target_encoding(X, X_test, y)
    X, X_test, keep_cols = select_features(X, X_test, y, spw, threshold_pct=10)

    # 2. 1차 학습
    print("\n" + "="*50)
    print("  1차 학습")
    print("="*50)
    lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test = kfold_train(
        X, X_test, y, spw, tag="[1차]"
    )
    opt_w1, opt_auc1 = optimize_weights(lgb_oof, xgb_oof, cat_oof, y)
    pred_1st = opt_w1[0]*lgb_test + opt_w1[1]*xgb_test + opt_w1[2]*cat_test

    # 3. Pseudo Labeling (0.95 / 0.05 — 성공/실패 균형 있게)
    confident_mask = (pred_1st >= 0.95) | (pred_1st <= 0.05)
    n_confident    = confident_mask.sum()
    print(f"\n🏷️  Pseudo Labeling (0.95/0.05)...")
    print(f"  확신도 높은 케이스: {n_confident}개")

    if n_confident > 0:
        X_pseudo   = X_test[confident_mask].copy()
        y_pseudo   = pd.Series(
            (pred_1st[confident_mask] >= 0.95).astype(int),
            name="임신 성공 여부"
        )
        X_pl = pd.concat([X, X_pseudo], ignore_index=True)
        y_pl = pd.concat([y, y_pseudo], ignore_index=True)
        spw_pl = (y_pl==0).sum() / (y_pl==1).sum()
        print(f"  성공 추가: {y_pseudo.sum()}개 / 실패 추가: {(y_pseudo==0).sum()}개")
        print(f"  학습 데이터: {len(X)}개 → {len(X_pl)}개")

        # 4. 2차 학습
        print("\n" + "="*50)
        print("  2차 학습 (Pseudo Labeling)")
        print("="*50)
        lgb_oof2, xgb_oof2, cat_oof2, lgb_test2, xgb_test2, cat_test2 = kfold_train(
            X_pl, X_test, y_pl, spw_pl, tag="[2차]"
        )
        opt_w2, opt_auc2 = optimize_weights(lgb_oof2, xgb_oof2, cat_oof2, y_pl)
        pred_2nd = opt_w2[0]*lgb_test2 + opt_w2[1]*xgb_test2 + opt_w2[2]*cat_test2
    else:
        print("  추가할 케이스 없음 → 1차 결과만 사용")
        pred_2nd  = pred_1st
        opt_auc2  = 0

    # 5. 기존 제출 파일 앙상블과 최종 합산
    print("\n🔀 기존 파일 + 새 예측 최종 앙상블...")
    old_files = [
        "submission_kfold.csv",
        "submission_fast.csv",
        "submission_fs.csv",
        "submission_v3.csv",
    ]
    old_preds = []
    for f in old_files:
        try:
            df_old = pd.read_csv(f)
            old_preds.append(df_old.iloc[:, 1].values)
            print(f"  ✅ {f} 로드")
        except:
            print(f"  ❌ {f} 없음")

    if old_preds:
        old_avg = np.mean(old_preds, axis=0)
        # 1차 + 2차 + 기존 앙상블 3자 혼합 (2차에 가중치 더 줌)
        if n_confident > 0:
            final_pred = 0.4*pred_2nd + 0.3*pred_1st + 0.3*old_avg
            print(f"\n  최종 혼합: 2차(40%) + 1차(30%) + 기존(30%)")
        else:
            final_pred = 0.5*pred_1st + 0.5*old_avg
            print(f"\n  최종 혼합: 1차(50%) + 기존(50%)")
    else:
        final_pred = pred_2nd if n_confident > 0 else pred_1st

    # 6. 저장
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_failure.csv", index=False)
    print(f"\n🏆 submission_failure.csv 저장 완료")
    print(f"   1차 OOF AUC: {opt_auc1:.5f}")
    if n_confident > 0:
        print(f"   2차 OOF AUC: {opt_auc2:.5f}")
