from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

DEFAULT_PIPELINE = ColumnTransformer(
    [
        (
            "Surname",
            Pipeline([("tfidf", TfidfVectorizer()), ("pca", TruncatedSVD(3))]),
            "Surname",
        ),
        (
            "SurGeoGendSal",
            Pipeline([("tfidf", TfidfVectorizer()), ("pca", TruncatedSVD(3))]),
            "SurGeoGendSal",
        ),
    ]
)


def data_pipeline(
    df: pd.DataFrame,
    is_train: bool = True,
    scaler: MinMaxScaler = MinMaxScaler(),
    encoder: Any = DEFAULT_PIPELINE,
) -> Tuple[pd.DataFrame, Any, MinMaxScaler]:
    df = engineer_features(df)
    df, encoder = encode_features(df, is_train, encoder)
    df = drop_features(df)
    df, scaler = standardize_features(df, is_train, scaler)
    return (df, encoder, scaler)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["SurnameLength"] = df["Surname"].apply(lambda x: len(x))
    df["AgeCategory"] = np.round(df.Age / 20).astype("int").astype("category")
    df["IsSenior"] = df["Age"].apply(lambda x: 1 if x >= 60 else 0)
    df["IsActiveByCrCard"] = df["HasCrCard"] * df["IsActiveMember"]
    df["ProductsPerTenure"] = df["Tenure"] / df["NumOfProducts"]
    df["SurGeoGendSal"] = (
        df["CustomerId"].astype("str")
        + df["Surname"]
        + df["Geography"]
        + df["Gender"]
        + np.round(df.EstimatedSalary).astype("str")
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
        "AgeCategory",
    ],
) -> Tuple[pd.DataFrame, Any]:
    if is_train:
        encoder.fit(df[["Surname", "SurGeoGendSal"]])

    df_encoded = pd.get_dummies(df[features], columns=features)
    df_encoded = df_encoded.apply(lambda x: x.astype("category").cat.codes)
    df = pd.concat([df, df_encoded], axis=1).drop(features, axis=1)

    name_components = pd.DataFrame(encoder.transform(df[["Surname", "SurGeoGendSal"]]))
    name_components.columns = [
        f"TextEmbedding{f}" for f in name_components.columns.tolist()
    ]
    df.reset_index(drop=True, inplace=True)
    name_components.reset_index(drop=True, inplace=True)
    df = pd.concat([df, name_components], axis=1)

    return (df, encoder)


def drop_features(
    df: pd.DataFrame,
    features: List = ["id", "CustomerId", "Surname", "SurGeoGendSal"],
):
    return df.drop(features, axis=1)


def standardize_features(
    df: pd.DataFrame,
    is_train: bool = True,
    scaler: MinMaxScaler = MinMaxScaler(),
    features: List = [
        "CreditScore",
        "Age",
        "Balance",
        "EstimatedSalary",
    ],
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    if len(features) == 0:
        return (df, scaler)

    if is_train:
        scaler.fit(df[features])

    df[features] = scaler.transform(df[features])
    return (df, scaler)
