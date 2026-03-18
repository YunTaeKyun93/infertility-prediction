import pandas as pd
import mlflow
import warnings
from sklearn.metrics import roc_auc_score
import os
import glob
import numpy as np

warnings.filterwarnings("ignore")

from preprocess import load_data, preprocess, apply_target_encoding,select_features
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

    X, X_test, y = preprocess(train_df, test_df) #type:ignore
    X, X_test = apply_target_encoding(X, X_test, y)
    spw = (y == 0).sum() / (y == 1).sum()
  
    lgb_best = tune_lgb(X, y, spw)
    xgb_best = tune_xgb(X, y, spw)
    cat_best = tune_cat(X, y, spw)

    lgb_oof, lgb_test, lgb_auc = train_lgb(X, X_test, y, lgb_best) #type:ignore
    xgb_oof, xgb_test, xgb_auc = train_xgb(X, X_test, y, xgb_best)
    cat_oof, cat_test, cat_auc = train_cat(X, X_test, y, cat_best)

    opt_w, opt_auc = optimize_weights(lgb_oof, xgb_oof, cat_oof, y)

    final_pred = opt_w[0]*lgb_test + opt_w[1]*xgb_test + opt_w[2]*cat_test
    final_pred = xgb_test

    good_files = [
    os.path.join(BASE_DIR, "../outputs/submissions/submission_v3.csv"),
    os.path.join(BASE_DIR, "../outputs/submissions/submission_fast.csv"),
    os.path.join(BASE_DIR, "../outputs/submissions/submission_v3_kfold_optuna.csv"),
    os.path.join(BASE_DIR, "../outputs/submissions/submission_kfold_optuna.csv"),
    os.path.join(BASE_DIR, "../outputs/submissions/submission_kfold.csv"),
    ]
    old_preds = []
    for f in good_files:
        df_old = pd.read_csv(f)
        old_preds.append(df_old["probability"].values)
        print(f"로드: {f}")

    if old_preds:
        old_avg = np.mean(old_preds, axis=0)
        final_pred = 0.5 * final_pred + 0.5 * old_avg
        print(f"기존 파일 {len(old_preds)}개 앙상블 적용")    
    
    confident_mask = (final_pred >= 0.95) | (final_pred <= 0.05)
    n_confident = confident_mask.sum()
    print(f"\n🏷️ Pseudo Labeling 확신도 높은 케이스: {n_confident}개")

    if n_confident > 0:
        X_pseudo = X_test[confident_mask].copy()
        y_pseudo = pd.Series(
            (final_pred[confident_mask] >= 0.95).astype(int),
            name="임신 성공 여부"
        )
        X_pl = pd.concat([X, X_pseudo], ignore_index=True)
        y_pl = pd.concat([y, y_pseudo], ignore_index=True)
        print(f"성공 추가: {y_pseudo.sum()}개 / 실패 추가: {(y_pseudo==0).sum()}개")
        print(f"학습 데이터: {len(X)}개 → {len(X_pl)}개")

    # 2차 학습
        spw_pl = (y_pl==0).sum() / (y_pl==1).sum()
        lgb_oof2, lgb_test2, _ = train_lgb(X_pl, X_test, y_pl, lgb_best)
        xgb_oof2, xgb_test2, _ = train_xgb(X_pl, X_test, y_pl, xgb_best)
        cat_oof2, cat_test2, _ = train_cat(X_pl, X_test, y_pl, cat_best)

        opt_w2, opt_auc2 = optimize_weights(lgb_oof2, xgb_oof2, cat_oof2, y_pl)
        pred_2nd = opt_w2[0]*lgb_test2 + opt_w2[1]*xgb_test2 + opt_w2[2]*cat_test2

    # 1차 + 2차 혼합
    final_pred = 0.6 * pred_2nd + 0.4 * final_pred
    print(f"2차 OOF AUC: {opt_auc2:.5f}")
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_tk.csv", index=False)
    print(f"\n submission_tk.csv 저장 완료  (OOF AUC: {opt_auc:.5f})")
    print(f"\n MLflow UI 확인: mlflow ui")

    
    print("\n 단독 vs 앙상블 비교")
    print(f"XGBoost 단독:        {roc_auc_score(y, xgb_oof):.5f}")
    print(f"XGB+CAT 앙상블:      {roc_auc_score(y, xgb_oof*0.7 + cat_oof*0.3):.5f}")
    print(f"최적가중치 앙상블:   {roc_auc_score(y, opt_w[0]*lgb_oof + opt_w[1]*xgb_oof + opt_w[2]*cat_oof):.5f}")



 