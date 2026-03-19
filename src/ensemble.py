import numpy as np
import mlflow
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize


def optimize_weights(lgb_oof, xgb_oof, cat_oof, y):
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
    print(f"  LightGBM 단독:     {lgb_auc:.5f}")
    print(f"  XGBoost  단독:     {xgb_auc:.5f}")
    print(f"  CatBoost 단독:     {cat_auc:.5f}")
    print(f"  균등 앙상블:       {equal_auc:.5f}")
    print(f"  최적가중치 앙상블: {opt_auc:.5f}  ← 최종 제출")
    print(f"  최적 가중치: LGB={opt_w[0]:.3f} / XGB={opt_w[1]:.3f} / CAT={opt_w[2]:.3f}")

    with mlflow.start_run(run_name="Ensemble"):
        mlflow.log_metric("lgb_oof_auc",   lgb_auc)
        mlflow.log_metric("xgb_oof_auc",   xgb_auc)
        mlflow.log_metric("cat_oof_auc",   cat_auc)
        mlflow.log_metric("equal_auc",     equal_auc)
        mlflow.log_metric("opt_auc",       opt_auc)
        mlflow.log_param("lgb_weight",     round(opt_w[0], 3))
        mlflow.log_param("xgb_weight",     round(opt_w[1], 3))
        mlflow.log_param("cat_weight",     round(opt_w[2], 3))

    return opt_w, opt_auc