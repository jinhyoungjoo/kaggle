from typing import Any, Dict

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier


def build_model() -> Any:
    params = get_parameters()

    lgbm_model = LGBMClassifier(**params["lgbm"])
    xgb_model = XGBClassifier(**params["xgb"])
    catboost_model = CatBoostClassifier(**params["cat"])

    model = VotingClassifier(
        [("LGBM", lgbm_model), ("XGB", xgb_model), ("CatBoost", catboost_model)],
        **params["vc"]
    )

    return model


def get_parameters() -> Dict:
    param_dict = {
        "lgbm": {
            "verbose": -1,
            "random_state": 503,
            "force_row_wise": True,
        },
        "xgb": {
            "verbosity": 0,
            "random_state": 503,
            "objective": "binary:logistic",
        },
        "cat": {
            "verbose": False,
            "random_state": 503,
        },
        "vc": {
            "voting": "soft",
            "n_jobs": -1,
            "verbose": False,
            "weights": [2.661069497099081, 3.108043849171256, 2.7211502982712887],
        },
    }

    optimization_results = {
        "lgbm__n_estimators": 747,
        "lgbm__subsample": 0.5874293048790149,
        "lgbm__colsample_bytree": 0.8916695198792095,
        "lgbm__learning_rate": 0.01238341029010065,
        "lgbm__max_depth": 6,
        "lgbm__num_leaves": 128,
        "lgbm__reg_alpha": 0.058511091986509,
        "lgbm__reg_lambda": 0.45056871760500017,
        "xgb__n_estimators": 1585,
        "xgb__learning_rate": 0.031181966200805735,
        "xgb__max_depth": 5,
        "xgb__subsample": 0.18178276589982506,
        "xgb__colsample_bytree": 0.8878002659372084,
        "xgb__min_child_weight": 14,
        "cat__iterations": 1171,
        "cat__learning_rate": 0.011665222117675678,
        "cat__depth": 9,
        "cat__subsample": 0.32517072561453486,
        "cat__colsample_bylevel": 0.5983827230216009,
        "cat__min_data_in_leaf": 70,
    }

    for key, value in optimization_results.items():
        model, param = [x.strip() for x in key.split("__")]
        param_dict[model][param] = value

    return param_dict
