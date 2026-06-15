import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def generate_synthetic_data(num_rows: int, seed: int) -> pd.DataFrame:
    """Generates synthetic heart disease data with a dependent target variable."""
    np.random.seed(seed)

    data = {
        "age": np.random.randint(29, 78, size=num_rows),
        "sex": np.random.choice(["male", "female"], size=num_rows, p=[0.68, 0.32]),
        "chest_pain_type": np.random.choice(
            ["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"],
            size=num_rows,
            p=[0.15, 0.25, 0.35, 0.25],
        ),
        "resting_blood_pressure": np.random.randint(94, 200, size=num_rows).astype(
            float
        ),
        "cholesterol": np.random.randint(126, 564, size=num_rows).astype(float),
        "fasting_blood_sugar": np.random.choice(
            [">120mg/dl", "<=120mg/dl"], size=num_rows, p=[0.15, 0.85]
        ),
        "resting_electrocardio": np.random.choice(
            ["normal", "st_t_wave_abnormality", "left_ventricular_hypertrophy"],
            size=num_rows,
            p=[0.5, 0.1, 0.4],
        ),
        "max_heart_rate": np.random.randint(71, 202, size=num_rows),
        "exercise_induced_angina": np.random.choice(
            ["yes", "no"], size=num_rows, p=[0.33, 0.67]
        ),
        "st_depression": np.round(np.random.uniform(0.0, 6.2, size=num_rows), 1),
        "slope": np.random.choice(
            ["upsloping", "flat", "downsloping"],
            size=num_rows,
            p=[0.45, 0.45, 0.10],
        ),
    }

    df = pd.DataFrame(data)

    # Risk factor log-odds calculation to ground predictive logic
    log_odds = (
        0.04 * (df["age"] - 50)
        + 0.005 * (df["cholesterol"] - 200)
        + 0.3 * (df["st_depression"])
        - 0.02 * (df["max_heart_rate"] - 150)
        + np.where(df["chest_pain_type"] == "asymptomatic", 1.2, -0.4)
        + np.where(df["exercise_induced_angina"] == "yes", 0.8, -0.2)
    )

    probabilities = 1 / (1 + np.exp(-log_odds))
    df["target"] = np.random.binomial(1, probabilities)

    # Missing value injection (~5% rate)
    for col in [
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar",
        "slope",
    ]:
        df.loc[np.random.rand(num_rows) < 0.05, col] = np.nan

    return df


def get_preprocessor() -> ColumnTransformer:
    """Creates the data preparation transformer strategy."""
    numeric_features = [
        "age",
        "resting_blood_pressure",
        "cholesterol",
        "max_heart_rate",
        "st_depression",
    ]
    categorical_features = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "resting_electrocardio",
        "exercise_induced_angina",
        "slope",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def prepare_data(config: dict):
    """Loads the DVC-tracked dataset and splits features and labels using configurations."""
    # Read the tracked dataset directly from the data folder
    df = pd.read_csv("data/heart_disease.csv")
    
    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_seed"],
        stratify=y,
    )



