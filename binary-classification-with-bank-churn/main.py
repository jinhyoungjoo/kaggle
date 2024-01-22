import argparse
import os

import pandas as pd
from model import build_model, optimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from utils import data_pipeline

DATA_DIR = "./data"


def main(args):
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    X, y = df_train.drop("Exited", axis=1), df_train["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=503
    )

    X_train, encoder, scaler = data_pipeline(X_train)  # type: ignore

    if args.optimize:
        optimize(X_train.values, y_train.values)  # type: ignore
        return

    X_test, _, _ = data_pipeline(
        X_test,  # type: ignore
        is_train=False,
        encoder=encoder,
        scaler=scaler,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print("Accuracy:", model.score(X_test, y_test))
    print("ROC-AUC Score:", roc_auc_score(y_test, predicted))
    print(confusion_matrix(y_test, predicted))

    # Generate final results for submission.
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X, id = df_test, df_test["id"]
    X, _, _ = data_pipeline(
        X,  # type: ignore
        is_train=False,
        encoder=encoder,
        scaler=scaler,
    )

    pd.DataFrame({"id": id.tolist(), "Exited": model.predict(X).tolist()}).to_csv(
        "./submission.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--optimize", help="Optimize the hyperparameters.", const=True, nargs="?"
    )
    args = parser.parse_args()
    main(args)
