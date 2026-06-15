import os
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier

# Explicit package imports
from src.data_preprocessing import get_preprocessor
from src.train import load_config

# =====================================================================
# FIXTURES (Shared Setup Objects)
# =====================================================================
@pytest.fixture
def sample_config():
    """Loads project settings from the centralized configuration file."""
    return load_config("configs/config.yaml")

@pytest.fixture
def real_dataset():
    """Loads the actual DVC-tracked dataset file from disk."""
    data_path = "data/heart_disease.csv"
    if not os.path.exists(data_path):
        pytest.skip("Dataset file not available. Run data generation first.")
    return pd.read_csv(data_path)

@pytest.fixture
def dummy_raw_data():
    """Generates a small, messy dataframe to test preprocessing edge cases."""
    # Using arrays filled explicitly with valid data to prevent syntax parser gaps
    ages_list = [45, 54, 61]
    heart_rates_list = [142, 160, 115]
    
    return pd.DataFrame({
        "age": ages_list,
        "sex": ["male", "female", "male"],
        "chest_pain_type": ["typical_angina", "asymptomatic", "non_anginal"],
        "resting_blood_pressure": [120.0, np.nan, 140.0],  
        "cholesterol": [210.0, 240.0, np.nan],            
        "fasting_blood_sugar": ["<=120mg/dl", ">120mg/dl", np.nan], 
        "resting_electrocardio": ["normal", "normal", "left_ventricular_hypertrophy"],
        "max_heart_rate": heart_rates_list,
        "exercise_induced_angina": ["no", "yes", "no"],
        "st_depression": [1.2, 0.0, 2.5],
        "slope": ["flat", np.nan, "upsloping"]             
    })

# =====================================================================
# LEVEL 1: UNIT TESTS FOR PREPROCESSING FUNCTIONS (6 Tests Required)
# =====================================================================

def test_preprocessing_handles_missing_numerical_values(dummy_raw_data):
    """1. Verifies that numerical missing values are successfully imputed (no NaNs left)."""
    preprocessor = get_preprocessor()
    transformed = preprocessor.fit_transform(dummy_raw_data)
    assert not np.isnan(transformed).any()

def test_preprocessing_handles_missing_categorical_values(dummy_raw_data):
    """2. Verifies that categorical missing values are imputed and don't break the encoder."""
    preprocessor = get_preprocessor()
    transformed = preprocessor.fit_transform(dummy_raw_data)
    assert len(transformed) == 3  

def test_preprocessing_encodes_categorical_variables(dummy_raw_data):
    """3. Verifies that text categories are converted into numeric one-hot encoded columns."""
    preprocessor = get_preprocessor()
    transformed = preprocessor.fit_transform(dummy_raw_data)
    assert transformed.shape[1] > len(dummy_raw_data.columns)

def test_preprocessing_does_not_modify_original_dataframe(dummy_raw_data):
    """4. Verifies that the fit_transform function treats the input data as immutable."""
    original_copy = dummy_raw_data.copy()
    preprocessor = get_preprocessor()
    _ = preprocessor.fit_transform(dummy_raw_data)
    pd.testing.assert_frame_equal(dummy_raw_data, original_copy)

def test_preprocessing_raises_error_for_empty_dataframe():
    """5. Verifies that passing an empty dataframe crashes cleanly with a ValueError."""
    preprocessor = get_preprocessor()
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        preprocessor.fit_transform(empty_df)

def test_preprocessing_raises_error_for_missing_required_columns(dummy_raw_data):
    """6. Verifies that an error is raised if a required feature column is missing completely."""
    broken_data = dummy_raw_data.drop(columns=["age"])
    preprocessor = get_preprocessor()
    preprocessor.fit(dummy_raw_data)
    # Catch Scikit-Learn's built-in missing column ValueError cleanly
    with pytest.raises(ValueError):
        preprocessor.transform(broken_data)


# =====================================================================
# LEVEL 2: DATA VALIDATION TESTS (3 Tests Required)
# =====================================================================

def test_dataset_contains_all_expected_columns(real_dataset):
    """1. Verifies that the actual tracking dataset contains all 12 mandatory columns."""
    expected_cols = {
        "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
        "fasting_blood_sugar", "resting_electrocardio", "max_heart_rate",
        "exercise_induced_angina", "st_depression", "slope", "target"
    }
    assert expected_cols.issubset(real_dataset.columns)

def test_dataset_target_variable_contains_only_valid_classes(real_dataset):
    """2. Verifies that the target contains only valid binary classification outputs (0 or 1)."""
    unique_targets = set(real_dataset["target"].unique())
    assert unique_targets.issubset({0, 1})

def test_dataset_numeric_features_fall_within_allowed_ranges(real_dataset):
    """3. Verifies that biological values (like age) are within realistic non-negative thresholds."""
    assert real_dataset["age"].min() >= 0
    assert real_dataset["age"].max() <= 120
    assert real_dataset["max_heart_rate"].min() > 0

# =====================================================================
# LEVEL 3: MODEL VALIDATION TESTS (2 Tests Required)
# =====================================================================

def test_model_produces_correct_prediction_type_and_shape(real_dataset):
    """1. Verifies that model inference generates the correct array format and shape."""
    preprocessor = get_preprocessor()
    X = real_dataset.drop(columns=["target"])
    y = real_dataset["target"]
    
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", AdaBoostClassifier(n_estimators=10))])
    pipeline.fit(X, y)
    
    predictions = pipeline.predict(X.head(5))
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 5

def test_model_achieves_minimum_performance_threshold(real_dataset):
    """2. Verifies that the classifier performance achieves a minimum baseline accuracy of 60%."""
    preprocessor = get_preprocessor()
    X = real_dataset.drop(columns=["target"])
    y = real_dataset["target"]
    
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", AdaBoostClassifier(random_state=42, n_estimators=50))])
    pipeline.fit(X, y)
    
    train_accuracy = pipeline.score(X, y)
    assert train_accuracy >= 0.60
