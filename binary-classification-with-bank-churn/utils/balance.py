from typing import Any

import pandas as pd
from xgboost import XGBRegressor


def predict_balance(df: pd.DataFrame) -> Any:
    df = df.loc[df["Balance"] != 0]
    X, y = df.drop("Balance", axis=1), df["Balance"]

    model = XGBRegressor()
    model.fit(X, y)
    return model
