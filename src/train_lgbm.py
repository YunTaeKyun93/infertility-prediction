import numpy as np
import lightgbm as lgb
import mlflow
import os
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
N_TRIALS   = 50
TUNE_FOLDS = 3
N_FOLDS    = 5
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(BASE_DIR, "outputs/figures/shap_importance.png")

def tune_lgb(X, y, spw):
    print(f"\nLightGBM Optuna 튜닝 ({N_TRIALS}  × {TUNE_FOLDS})...")
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
            aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))#type: ignore
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True) #type: ignore
    
    print(f"  ➜ LGB 최적 AUC: {study.best_value:.5f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


def train_lgb(X, X_test, y, lgb_best):
    spw = (y == 0).sum() / (y == 1).sum()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lgb_oof  = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))

    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "verbose": -1, "n_jobs": -1, "random_state": SEED,
        "scale_pos_weight": spw, "n_estimators": 3000,
        **(lgb_best or {"learning_rate": 0.05, "num_leaves": 127}),
    }

    with mlflow.start_run(run_name="LightGBM"):
        mlflow.log_params(lgb_params)

        print(f"\n LGBM {N_FOLDS}-Fold 학습...주우우웅")
        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            m = lgb.LGBMClassifier(**lgb_params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
            lgb_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1] #type: ignore
            lgb_test += m.predict_proba(X_test)[:, 1] / N_FOLDS #type: ignore

            fold_auc = roc_auc_score(y.iloc[val_idx], lgb_oof[val_idx])
            fold_aucs.append(fold_auc)
            mlflow.log_metric(f"fold_{fold+1}_auc", fold_auc) #type:ignore
            print(f"  Fold {fold+1} AUC: {fold_auc:.5f}")

        oof_auc = roc_auc_score(y, lgb_oof)
        mlflow.log_metric("oof_auc", oof_auc)#type:ignore
        print(f"  ➜ LGB OOF AUC: {oof_auc:.5f}")

    print("SHA계산 합니당")
    explainer = shap.TreeExplainer(m)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap.summary_plot(shap_vals[1], X, max_display=30, show=False)
    else:
        shap.summary_plot(shap_vals, X, max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP 저장 완료 → outputs/figures/shap_importance.png")

    return lgb_oof, lgb_test, oof_auc