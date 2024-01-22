from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .balance import predict_balance


def data_pipeline(
    df: pd.DataFrame,
    is_train: bool = True,
    scaler: MinMaxScaler = MinMaxScaler(),
    encoder: Pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("pca", TruncatedSVD(3))]
    ),
) -> Tuple[pd.DataFrame, Pipeline, MinMaxScaler]:
    df = engineer_features(df)
    df, encoder = encode_features(df, is_train, encoder)
    df = drop_features(df)
    df, scaler = standardize_features(df, is_train, scaler)
    return (df, encoder, scaler)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["SurnameLength"] = df["Surname"].apply(lambda x: len(x))
    df["SurnameFirstLetter"] = df["Surname"].apply(lambda x: x[0])

    df["IsSenior"] = df["Age"].apply(lambda x: 1 if x >= 60 else 0)
    df["AgeNumOfProducts"] = df["Age"] ** df["NumOfProducts"]
    df["IsActiveByCrCard"] = df["HasCrCard"] * df["IsActiveMember"]
    df["ProductsPerTenure"] = df["Tenure"] / df["NumOfProducts"]

    df["AgeCategory"] = np.round(df["Age"] / 10).astype("int").astype("category")
    df["BalanceToSalary"] = df["Balance"] / df["EstimatedSalary"]
    df["CreditScoreToAge"] = df["CreditScore"] / df["Age"]

    df["BalanceToProduct"] = df["Balance"] / df["NumOfProducts"]
    df["TenureToAgeRatio"] = df["Tenure"] / df["Age"]
    df["CreditScoreCategory"] = pd.cut(
        df["CreditScore"], bins=[0, 650, 750, 850], labels=["Low", "Medium", "High"]
    )

    balance_bins = [-np.inf, 0, 10000, 50000, 100000, np.inf]
    df["BalanceCategory"] = pd.cut(
        df["Balance"], bins=balance_bins, labels=False, right=False
    )

    return df


def encode_features(
    df: pd.DataFrame,
    is_train: bool,
    encoder: Pipeline,
    features: List = [
        "Geography",
        "Gender",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "AgeCategory",
        "IsSenior",
        "IsActiveByCrCard",
        "BalanceCategory",
        "CreditScoreCategory",
        "SurnameFirstLetter",
    ],
) -> Tuple[pd.DataFrame, Any]:
    if is_train:
        encoder.fit(df["Surname"])

    df_encoded = pd.get_dummies(df[features], columns=features)
    df_encoded = df_encoded.apply(lambda x: x.astype("category").cat.codes)
    df = pd.concat([df, df_encoded], axis=1).drop(features, axis=1)

    name_components = pd.DataFrame(encoder.transform(df["Surname"]))
    name_components.columns = [f"Name_{f}" for f in name_components.columns.tolist()]
    df.reset_index(drop=True, inplace=True)
    name_components.reset_index(drop=True, inplace=True)
    df = pd.concat([df, name_components], axis=1)

    return (df, encoder)


def drop_features(
    df: pd.DataFrame,
    features: List = [
        "id",
        "CustomerId",
        "Surname",
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
    scaler: MinMaxScaler = MinMaxScaler(),
    features: List = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "EstimatedSalary",
        "SurnameLength",
        "ProductsPerTenure",
        "BalanceToSalary",
        "CreditScoreToAge",
        "AgeNumOfProducts",
        "BalanceToProduct",
        "TenureToAgeRatio",
    ],
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    if is_train:
        scaler.fit(df[features])

    df[features] = scaler.transform(df[features])
    return (df, scaler)
