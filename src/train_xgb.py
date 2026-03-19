import numpy as np
import xgboost as xgb
import mlflow
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
N_TRIALS   = 50
TUNE_FOLDS = 3
N_FOLDS    = 5

def tune_xgb(X, y, spw):
    print(f"\nXGBoost Optuna 튜닝 ({N_TRIALS}  × {TUNE_FOLDS})...")

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
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)#type:ignore
    print(f"  ➜ XGB 최적 AUC: {study.best_value:.5f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


def train_xgb(X, X_test, y, xgb_best):
    spw = (y == 0).sum() / (y == 1).sum()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    xgb_oof  = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "verbosity": 0,
        "n_jobs": -1, "random_state": SEED,
        "scale_pos_weight": spw, "n_estimators": 3000,
        "early_stopping_rounds": 100,
        **(xgb_best or {"learning_rate": 0.05, "max_depth": 6}),
    }

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_params(xgb_params)

        print(f"\n⚡ XGBoost {N_FOLDS}-Fold 학습...")
        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            m = xgb.XGBClassifier(**xgb_params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  verbose=False)
            xgb_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
            xgb_test += m.predict_proba(X_test)[:, 1] / N_FOLDS

            fold_auc = roc_auc_score(y.iloc[val_idx], xgb_oof[val_idx])
            fold_aucs.append(fold_auc)
            mlflow.log_metric(f"fold_{fold+1}_auc", fold_auc)#type:ignore
            print(f"  Fold {fold+1} AUC: {fold_auc:.5f}")

        oof_auc = roc_auc_score(y, xgb_oof)
        mlflow.log_metric("oof_auc", oof_auc)#type:ignore
        print(f"  ➜ XGB OOF AUC: {oof_auc:.5f}")

    return xgb_oof, xgb_test, oof_auc