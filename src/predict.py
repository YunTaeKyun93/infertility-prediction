import pandas as pd
import mlflow
import warnings
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import os
import numpy as np

warnings.filterwarnings("ignore")

from preprocess import load_data, preprocess, apply_target_encoding
from train_lgbm import tune_lgb, train_lgb
from train_xgb  import tune_xgb, train_xgb
from train_cat  import tune_cat, train_cat
from ensemble   import optimize_weights


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "../data/train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "../data/test.csv")
SAVE_DIR   = os.path.join(BASE_DIR, "../outputs/submissions")
EXPERIMENT = "난임_예측"
mlflow.set_tracking_uri(os.path.join(BASE_DIR, "../mlruns"))


if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT)

    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    X, X_test, y = preprocess(train_df, test_df)  # type:ignore
    X, X_test = apply_target_encoding(X, X_test, y)
    spw = (y == 0).sum() / (y == 1).sum()

    lgb_best = tune_lgb(X, y, spw)
    xgb_best = tune_xgb(X, y, spw)
    cat_best = tune_cat(X, y, spw)

    lgb_oof, lgb_test, lgb_auc = train_lgb(X, X_test, y, lgb_best)  # type:ignore
    xgb_oof, xgb_test, xgb_auc = train_xgb(X, X_test, y, xgb_best)
    cat_oof, cat_test, cat_auc = train_cat(X, X_test, y, cat_best)

    opt_w, opt_auc = optimize_weights(lgb_oof, xgb_oof, cat_oof, y)

    # GBDT 최적 앙상블
    final_pred = opt_w[0]*lgb_test + opt_w[1]*xgb_test + opt_w[2]*cat_test
    print(f"\nGBDT 최적가중치 앙상블 OOF AUC: {opt_auc:.5f}")

    # 기존 제출 파일 앙상블
    good_files = [
        os.path.join(BASE_DIR, "../outputs/submissions/submission_v3.csv"),
        os.path.join(BASE_DIR, "../outputs/submissions/submission_fast.csv"),
        os.path.join(BASE_DIR, "../outputs/submissions/submission_v3_kfold_optuna.csv"),
        os.path.join(BASE_DIR, "../outputs/submissions/submission_kfold_optuna.csv"),
        os.path.join(BASE_DIR, "../outputs/submissions/submission_kfold.csv"),
    ]
    old_preds = []
    for f in good_files:
        try:
            df_old = pd.read_csv(f)
            old_preds.append(df_old["probability"].values)
            print(f"로드: {f}")
        except:
            print(f"없음: {f}")

    if old_preds:
        old_avg = np.mean(old_preds, axis=0)
        final_pred = 0.5 * final_pred + 0.5 * old_avg
        print(f"기존 파일 {len(old_preds)}개 앙상블 적용")

    # MLP 앙상블
    mlp_path = os.path.join(BASE_DIR, "../outputs/submissions/submission_mlp.csv")
    try:
        mlp_pred = pd.read_csv(mlp_path)["probability"].values
        final_pred = 0.8 * final_pred + 0.2 * mlp_pred
        print(f"MLP 앙상블 적용 (GBDT 80% + MLP 20%)")
    except:
        print(f"MLP 파일 없음 → GBDT만 사용")

    # 저장
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_tk.csv", index=False)
    print(f"\nsubmission_tk.csv 저장 완료  (OOF AUC: {opt_auc:.5f})")
    print(f"\nMLflow UI 확인: mlflow ui --port 5001")

    print("\n단독 vs 앙상블 비교")
    print(f"LightGBM 단독:       {roc_auc_score(y, lgb_oof):.5f}")
    print(f"XGBoost 단독:        {roc_auc_score(y, xgb_oof):.5f}")
    print(f"CatBoost 단독:       {roc_auc_score(y, cat_oof):.5f}")
    print(f"최적가중치 앙상블:   {roc_auc_score(y, opt_w[0]*lgb_oof + opt_w[1]*xgb_oof + opt_w[2]*cat_oof):.5f}")



    