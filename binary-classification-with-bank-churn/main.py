import os
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "./data"


def main():
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    X, y = df_train.drop("Exited", axis=1), df_train["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = encode_features(X_train)  # type: ignore
    X_train = drop_features(X_train)
    X_train, standard_scalar = standardize_features(X_train)

    X_test = encode_features(X_test)  # type: ignore
    X_test = drop_features(X_test)
    X_test, standard_scalar = standardize_features(
        X_test, is_train=False, standard_scalar=standard_scalar
    )

    model = RandomForestClassifier()
    model = model.fit(X_train, y_train)
    print("Accuracy:", model.score(X_test, y_test))

    # Generate final results for submission.
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X, id = df_test, df_test["id"]
    X = encode_features(X)
    X = drop_features(X)
    X, _ = standardize_features(X, is_train=False, standard_scalar=standard_scalar)
    pd.DataFrame({"id": id.tolist(), "Exited": model.predict(X).tolist()}).to_csv(
        "./submission.csv", index=False
    )


def drop_features(
    df: pd.DataFrame,
    features: List = [
        "id",
        "CustomerId",
        "Surname",
        "Geography",
        "Gender",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ],
):
    return df.drop(features, axis=1)


def encode_features(
    df: pd.DataFrame,
    features: List = [
        "Geography",
        "Gender",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ],
) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df[features], columns=features)
    return pd.concat([df, df_encoded], axis=1)


def standardize_features(
    df: pd.DataFrame,
    is_train: bool = True,
    standard_scalar: StandardScaler = StandardScaler(),
) -> Tuple[pd.DataFrame, StandardScaler]:
    if is_train:
        standard_scalar.fit(df)

    df = pd.DataFrame(standard_scalar.transform(df), columns=df.columns)
    return (df, standard_scalar)


if __name__ == "__main__":
    main()
