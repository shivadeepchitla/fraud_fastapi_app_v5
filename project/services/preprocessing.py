
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline(numeric_cols, ordinal_cols, one_hot_cols):

    # ---------------------------
    # Preprocessing Pipelines
    # ---------------------------

    # numeric_transformer = Pipeline([
    #     ("imputer", SimpleImputer(strategy="median")),
    #     ("scaler", StandardScaler())
    # ])

    one_hot_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("onehot", one_hot_transformer, one_hot_cols),
        ("ordinal", ordinal_transformer, ordinal_cols)
    ])

    return preprocessor
