from functools import partial

import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def optimize(X, y) -> None:
    def _optimize(trial, X, y):
        lgbm_model = LGBMClassifier(**build_lgbm_params(trial))
        xgb_model = XGBClassifier(**build_xgb_params(trial))
        catboost_model = CatBoostClassifier(**build_catboost_params(trial))

        model = VotingClassifier(
            [("LGBM", lgbm_model), ("XGB", xgb_model), ("CatBoost", catboost_model)],
            voting="soft",
            n_jobs=-1,
            verbose=False,
            weights=[
                trial.suggest_float("vc__lgbm_weight", 0.1, 5.0),
                trial.suggest_float("vc__xgb_weight", 0.1, 5.0),
                trial.suggest_float("vc__cat_weight", 0.1, 5.0),
            ],
        )

        kfold, scores = StratifiedKFold(n_splits=5), []
        for train_idx, val_idx in kfold.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model.fit(X_train, y_train)
            scores.append(roc_auc_score(y_val, model.predict(X_val)))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(_optimize, X=X, y=y),  # type: ignore
        n_trials=200,
        n_jobs=-1,
        show_progress_bar=True,
    )


def build_lgbm_params(trial):
    lgbm_params = {
        "verbose": -1,
        "n_estimators": trial.suggest_int("lgbm__n_estimators", 500, 2500),
        "subsample": trial.suggest_float("lgbm__subsample", 0.05, 1),
        "colsample_bytree": trial.suggest_float("lgbm__colsample_bytree", 0.05, 1.0),
        "learning_rate": trial.suggest_float("lgbm__learning_rate", 1e-3, 0.1),
        "max_depth": trial.suggest_int("lgbm__max_depth", 1, 10),
        "num_leaves": trial.suggest_int("lgbm__num_leaves", 1, 1000),
        "reg_alpha": trial.suggest_float("lgbm__reg_alpha", 0.05, 1),
        "reg_lambda": trial.suggest_float("lgbm__reg_lambda", 0.05, 1),
        "random_state": 503,
        "force_row_wise": True,
    }

    return lgbm_params


def build_xgb_params(trial):
    xgb_params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "n_estimators": trial.suggest_int("xgb__n_estimators", 500, 2500),
        "learning_rate": trial.suggest_float("xgb__learning_rate", 1e-3, 0.1),
        "max_depth": trial.suggest_int("xgb__max_depth", 1, 10),
        "subsample": trial.suggest_float("xgb__subsample", 0.05, 1),
        "colsample_bytree": trial.suggest_float("xgb__colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("xgb__min_child_weight", 1, 20),
        "random_state": 503,
    }

    return xgb_params


def build_catboost_params(trial):
    catboost_params = {
        "verbose": False,
        "iterations": trial.suggest_int("cat__iterations", 500, 1500),
        "learning_rate": trial.suggest_float("cat__learning_rate", 1e-3, 0.1),
        "depth": trial.suggest_int("cat__depth", 1, 10),
        "subsample": trial.suggest_float("cat__subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("cat__colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("cat__min_data_in_leaf", 1, 100),
        "random_state": 503,
    }

    return catboost_params
