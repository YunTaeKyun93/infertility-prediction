import numpy as np
import mlflow
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
N_TRIALS   = 50
TUNE_FOLDS = 3
N_FOLDS    = 5


def tune_cat(X, y, spw):
    print(f"\nCatBoost Optuna 튜닝 ({N_TRIALS}  × {TUNE_FOLDS})...")

    def objective(trial):
        params = {
            "eval_metric": "Logloss", "od_type": "Iter", "od_wait": 50,
            "verbose": False, "random_seed": SEED, "thread_count": -1,
            "scale_pos_weight": spw, "iterations": 2000,
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":               trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":     trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count":        trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 50),
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


def train_cat(X, X_test, y, cat_best):
    spw = (y == 0).sum() / (y == 1).sum()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cat_oof  = np.zeros(len(X))
    cat_test = np.zeros(len(X_test))

    cat_params = {
        "eval_metric": "Logloss", "od_type": "Iter", "od_wait": 100,
        "verbose": False, "random_seed": SEED, "thread_count": -1,
        "scale_pos_weight": spw, "iterations": 3000,
        **(cat_best or {"learning_rate": 0.05, "depth": 6}),
    }

    with mlflow.start_run(run_name="CatBoost"):
        mlflow.log_params(cat_params)

        print(f"\nCatBoost {N_FOLDS}-Fold 학습...")
        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            m = CatBoostClassifier(**cat_params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                  use_best_model=True)
            cat_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
            cat_test += m.predict_proba(X_test)[:, 1] / N_FOLDS

            fold_auc = roc_auc_score(y.iloc[val_idx], cat_oof[val_idx])
            fold_aucs.append(fold_auc)
            mlflow.log_metric(f"fold_{fold+1}_auc", fold_auc)
            print(f"  Fold {fold+1} AUC: {fold_auc:.5f}")

        oof_auc = roc_auc_score(y, cat_oof)
        mlflow.log_metric("oof_auc", oof_auc)
        print(f"  ➜ CAT OOF AUC: {oof_auc:.5f}")

    return cat_oof, cat_test, oof_auc