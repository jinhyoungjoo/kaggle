from typing import Any, Dict, Union

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier


def build_model(hyperparameters: Union[str, None]) -> Any:
    params = get_parameters(hyperparameters)

    lgbm_model = LGBMClassifier(**params["lgbm"])
    xgb_model = XGBClassifier(**params["xgb"])
    catboost_model = CatBoostClassifier(**params["cat"])

    model = VotingClassifier(
        [("LGBM", lgbm_model), ("XGB", xgb_model), ("CatBoost", catboost_model)],
        **params["vc"]
    )

    return model


def get_parameters(hyperparameters: Union[str, None]) -> Dict:
    param_dict = {
        "lgbm": {
            "verbose": -1,
            "random_state": 503,
            "force_row_wise": True,
            "n_estimators": 960,
            "subsample": 0.4604756677696442,
            "colsample_bytree": 0.465375841230126,
            "learning_rate": 0.04531704129811222,
            "max_depth": 6,
            "num_leaves": 803,
            "reg_alpha": 0.4132098753297654,
            "reg_lambda": 0.9992674466487466,
        },
        "xgb": {
            "verbosity": 0,
            "random_state": 503,
            "objective": "binary:logistic",
            "n_estimators": 973,
            "learning_rat e": 0.0786311558099196,
            "max_depth": 9,
            "subsample": 0.6451633803299511,
            "colsample_bytree": 0.20800229296622322,
            "min_child_weight": 11,
        },
        "cat": {
            "verbose": False,
            "random_state": 503,
            "iterations": 1395,
            "learning_rate": 0.025092883785253036,
            "depth": 6,
            "subsample": 0.32039321811073496,
            "colsample_bylevel": 0.47765607925544096,
            "min_data_in_leaf": 30,
        },
        "vc": {
            "voting": "soft",
            "n_jobs": -1,
            "verbose": False,
            "weights": [3.0146183474429327, 0.5762900154092979, 1.5605382149604368],
        },
    }

    if hyperparameters is not None:
        optimization_results = joblib.load(hyperparameters).best_trial.params
        print(optimization_results)

        for key, value in optimization_results.items():
            model, param = [x.strip() for x in key.split("__")]
            param_dict[model][param] = value

    return param_dict
