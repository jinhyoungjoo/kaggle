from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .balance import predict_balance


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


def remove_outliers(
    df: pd.DataFrame, is_train: bool = True, params_dict: Dict = {}
) -> Tuple[pd.DataFrame, Dict]:
    if is_train:
        model = predict_balance(df)
        params_dict["BalancePredictor"] = model

    balance = np.array(df["Balance"].tolist())
    predicted_balance = params_dict["BalancePredictor"].predict(
        df.drop("Balance", axis=1)
    )
    balance = np.where(balance != 0, balance, predicted_balance)

    return (df, params_dict)


def standardize_features(
    df: pd.DataFrame,
    is_train: bool = True,
    standard_scalar: StandardScaler = StandardScaler(),
) -> Tuple[pd.DataFrame, StandardScaler]:
    if is_train:
        standard_scalar.fit(df)

    df = pd.DataFrame(standard_scalar.transform(df), columns=df.columns)
    return (df, standard_scalar)
