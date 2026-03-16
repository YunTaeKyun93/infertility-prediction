import pandas as pd
import mlflow
import warnings
import os
warnings.filterwarnings("ignore")


from preprocess import load_data, preprocess
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

    X, X_test, y = preprocess(train_df, test_df)
    spw = (y == 0).sum() / (y == 1).sum()

    lgb_best = tune_lgb(X, y, spw)
    xgb_best = tune_xgb(X, y, spw)
    cat_best = tune_cat(X, y, spw)

    lgb_oof, lgb_test, lgb_auc = train_lgb(X, X_test, y, lgb_best)
    xgb_oof, xgb_test, xgb_auc = train_xgb(X, X_test, y, xgb_best)
    cat_oof, cat_test, cat_auc = train_cat(X, X_test, y, cat_best)

    opt_w, opt_auc = optimize_weights(lgb_oof, xgb_oof, cat_oof, y)

    final_pred = opt_w[0]*lgb_test + opt_w[1]*xgb_test + opt_w[2]*cat_test
    submission = pd.DataFrame({"ID": test_df["ID"], "probability": final_pred})
    submission.to_csv(f"{SAVE_DIR}/submission_v3.csv", index=False)
    print(f"\n submission_tk.csv 저장 완료  (OOF AUC: {opt_auc:.5f})")
    print(f"\n MLflow UI 확인: mlflow ui")