import argparse
import os
from typing import Union

import numpy as np
import pandas as pd
from model import build_model, optimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from utils import data_pipeline

DATA_DIR = "./data"
OPTUNA_STUDY_DIR = "./optuna_results.pkl"


def main(args):
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X, y = df_train.drop("Exited", axis=1), df_train["Exited"]
    X_test = df_test

    if args.optimize:
        X, _, _ = data_pipeline(X)
        optimize(X.values, y.values, OPTUNA_STUDY_DIR)
        kfold_prediction(X, y, X_test, hyperparameters=OPTUNA_STUDY_DIR).to_csv(
            "./submission.csv", index=False
        )
        return

    kfold_prediction(X, y, X_test).to_csv("./submission.csv", index=False)


def kfold_prediction(
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    num_folds: int = 5,
    hyperparameters: Union[str, None] = None,
) -> pd.DataFrame:
    pd.options.mode.copy_on_write = True

    kfolds, scores = StratifiedKFold(n_splits=num_folds), []
    test_predictions = np.empty((num_folds, len(X_test)))

    for fold_idx, (train_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        X_train, encoder, scaler = data_pipeline(X_train)
        X_val, _, _ = data_pipeline(
            X_val,
            is_train=False,
            encoder=encoder,
            scaler=scaler,
        )

        model = build_model(hyperparameters)
        model.fit(X_train, y_train)
        predicted = model.predict_proba(X_val)[:, 1]
        print(f"Accuracy (Fold {fold_idx}): {model.score(X_val, y_val)}")
        print(f"ROC-AUC Score (Fold {fold_idx}): {roc_auc_score(y_val, predicted)}")

        test, _, _ = data_pipeline(
            X_test,
            is_train=False,
            encoder=encoder,
            scaler=scaler,
        )

        scores.append(roc_auc_score(y_val, predicted))
        test_predictions[fold_idx, :] = model.predict_proba(test)[:, 1]

    y_pred = test_predictions.mean(axis=0)
    print(f"Average ROC-AUC Score: {np.mean(scores)}")

    df_final = X_test[["id"]]
    df_final["Exited"] = y_pred
    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize", help="Optimize the hyperparameters.", const=True, nargs="?"
    )
    args = parser.parse_args()
    main(args)
