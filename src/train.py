import os
import pickle
import sys
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Relative module imports
from src.data_preprocessing import get_preprocessor, prepare_data
from src.evaluation import evaluate_and_save_metrics

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_dvc_hash(dvc_path: str) -> str:
    """Extracts the unique MD5 data version hash from the DVC file metadata."""
    try:
        with open(dvc_path, "r") as f:
            dvc_data = yaml.safe_load(f)
            return dvc_data["outs"][0]["md5"]
    except Exception:
        return "unknown_or_local_simulation"


def run_training_pipeline():
    """Executes the pipeline, dynamically switching between local MLflow storage and CI modes."""
    config = load_config()
    
    # Check if the execution context is a remote GitHub Actions container
    is_ci_environment = os.environ.get("GITHUB_ACTIONS") == "true"

    if not is_ci_environment:
        # Single database connection configuration for stable local execution
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        # Configure the active MLflow Experiment space
        mlflow.set_experiment("Heart_Disease_Prediction_Tracking")
        mlflow.start_run()

    # --- CORE WORKFLOW LOGIC (Always Runs in Local and CI Contexts) ---
    
    # 1. Fetch and track versioning parameters
    dvc_hash = get_dvc_hash(config["paths"]["dvc_pointer"])
    if not is_ci_environment:
        mlflow.log_param("dvc_data_version", dvc_hash)

    # 2. Extract and register hyperparameters
    if not is_ci_environment:
        for param, value in config["model_params"].items():
            mlflow.log_param(param, value)

    # Prepare datasets
    X_train, X_test, y_train, y_test = prepare_data(config)
    preprocessor = get_preprocessor()

    # Choose algorithmic model dynamically based on configuration inputs
    model_type = config["model_params"].get("model_type", "AdaBoost")
    if model_type == "RandomForest":
        classifier = RandomForestClassifier(
            random_state=config["data"]["random_seed"],
            n_estimators=config["model_params"]["n_estimators"]
        )
    else:
        classifier = AdaBoostClassifier(
            random_state=config["data"]["random_seed"],
            n_estimators=config["model_params"]["n_estimators"],
            learning_rate=config["model_params"]["learning_rate"],
        )

    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    # Train model
    full_pipeline.fit(X_train, y_train)

    # Evaluate performance
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_test, y_proba))

    # Save local evaluation metrics files to disk via our evaluation module
    evaluate_and_save_metrics(y_test, y_pred, y_proba, config)

    # 3. Log evaluation metrics to MLflow server (Only if running locally)
    if not is_ci_environment:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", auc)
        print(f"Logged Run metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    else:
        print(f"CI Mode Active (Skipped Database Logs) -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # === CI PERFORMANCE GATEWAY THRESHOLD CHECK ===
    MIN_ACCURACY_THRESHOLD = 0.65
    if acc < MIN_ACCURACY_THRESHOLD:
        print(f"ERROR: Model accuracy ({acc:.4f}) dropped below mandatory CI gateway performance threshold ({MIN_ACCURACY_THRESHOLD})!")
        if not is_ci_environment:
            mlflow.end_run()
        sys.exit(1) # Exits with a non-zero code to break the GitHub Actions runner
    else:
        print("Success! Performance threshold verification passed.")

    # 4. Log the trained model as an MLflow artifact (Only if running locally)
    if not is_ci_environment:
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model_artifact",
            registered_model_name=f"HeartDisease_{model_type}"
        )
        mlflow.end_run()


if __name__ == "__main__":
    run_training_pipeline()
