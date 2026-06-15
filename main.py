import numpy as np
import pandas as pd
import json
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. SETUP & SIMULATE DATA (From Prior Step)
# ==========================================
import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 1200

# 1. Generate base features independently
data = {
    "age": np.random.randint(29, 78, size=n_rows),
    "sex": np.random.choice(["male", "female"], size=n_rows, p=[0.68, 0.32]),
    "chest_pain_type": np.random.choice(
        ["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"],
        size=n_rows,
        p=[0.15, 0.25, 0.35, 0.25],
    ),
    "resting_blood_pressure": np.random.randint(94, 200, size=n_rows).astype(
        float
    ),
    "cholesterol": np.random.randint(126, 564, size=n_rows).astype(float),
    "fasting_blood_sugar": np.random.choice(
        [">120mg/dl", "<=120mg/dl"], size=n_rows, p=[0.15, 0.85]
    ),
    "resting_electrocardio": np.random.choice(
        ["normal", "st_t_wave_abnormality", "left_ventricular_hypertrophy"],
        size=n_rows,
        p=[0.5, 0.1, 0.4],
    ),
    "max_heart_rate": np.random.randint(71, 202, size=n_rows),
    "exercise_induced_angina": np.random.choice(
        ["yes", "no"], size=n_rows, p=[0.33, 0.67]
    ),
    "st_depression": np.round(np.random.uniform(0.0, 6.2, size=n_rows), 1),
    "slope": np.random.choice(
        ["upsloping", "flat", "downsloping"], size=n_rows, p=[0.45, 0.45, 0.10]
    ),
}

df = pd.DataFrame(data)

# 2. Inject realistic feature weights to compute target probabilities
# Construct a log-odds equation where risk factors actively drive heart disease
log_odds = (
    0.04 * (df["age"] - 50)
    + 0.005 * (df["cholesterol"] - 200)
    + 0.3 * (df["st_depression"])
    - 0.02 * (df["max_heart_rate"] - 150)
    + np.where(df["chest_pain_type"] == "asymptomatic", 1.2, -0.4)
    + np.where(df["exercise_induced_angina"] == "yes", 0.8, -0.2)
)

# Convert log-odds into probabilities using the sigmoid function
probabilities = 1 / (1 + np.exp(-log_odds))

# Generate the target variable dynamically based on those probabilities
df["target"] = np.random.binomial(1, probabilities)

# 3. Simulate missing data structures (~5% missingness)
cols_with_nan = [
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "slope",
]
for col in cols_with_nan:
    nan_mask = np.random.rand(n_rows) < 0.05
    df.loc[nan_mask, col] = np.nan


# Split features and label
X = df.drop(columns=["target"])
y = df["target"]

# Create train/test splits (80/20 partition)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 2. DEFINE PREPROCESSING PIPELINES
# ==========================================
# Group columns by data type
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

# Numeric transformer: Median imputation followed by standard scaling
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Categorical transformer: Mode imputation followed by One-Hot encoding
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# Bundle both preprocessing streams together
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ==========================================
# 3. BUILD AND TRAIN FINAL FULL PIPELINE
# ==========================================
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(random_state=42, n_estimators=100)),
    ]
)

# Train the model (Preprocesses internally without data leakage!)
full_pipeline.fit(X_train, y_train)

# ==========================================
# 4. EVALUATE MODEL PERFORMANCE
# ==========================================
# Generate predictions
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

# Display evaluation summaries
print("\n--- Model Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


### Create metrics in json

# 1. Create the 'model' directory safely if it does not already exist
os.makedirs("models", exist_ok=True)

# 2. Define the storage path for your serialized model
model_path = os.path.join("models", "model.pkl")

# 3. Export the entire trained pipeline to disk
with open(model_path, "wb") as file:
    pickle.dump(full_pipeline, file)

print(f"Success! Model pipeline safely saved to: {model_path}")

# 1. Generate the classification report as a dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# 2. Append the ROC-AUC score to the dictionary
report_dict["roc_auc_score"] = float(roc_auc_score(y_test, y_proba))

# 3. Create the separate 'metrics' directory if it doesn't exist
os.makedirs("metrics", exist_ok=True)

# 4. Save the payload into its own folder
metrics_path = os.path.join("metrics", "metrics.json")
with open(metrics_path, "w") as json_file:
    json.dump(report_dict, json_file, indent=4)

print(f"Metrics successfully isolated and recorded at: {metrics_path}")

