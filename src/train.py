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
    """Executes the pipeline and records all artifacts/metrics to MLflow."""
    config = load_config()
    
    #single database connection line right at the very beginning of the function:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Configure the active MLflow Experiment space
    mlflow.set_experiment("Heart_Disease_Prediction_Tracking")

    with mlflow.start_run():
        # 1. Fetch data versioning pointers via DVC
        dvc_hash = get_dvc_hash(config["paths"]["dvc_pointer"])
        mlflow.log_param("dvc_data_version", dvc_hash)

        # 2. Log all hyperparameters from config
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

        # 3. Log 3 mandatory evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        auc = float(roc_auc_score(y_test, y_proba))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", auc)

        print(f"Logged Run metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # === NEW: CI PERFORMANCE GATEWAY THRESHOLD ===
        MIN_ACCURACY_THRESHOLD = 0.65
        if acc < MIN_ACCURACY_THRESHOLD:
            print(f"ERROR: Model accuracy ({acc:.4f}) dropped below mandatory CI gateway performance threshold ({MIN_ACCURACY_THRESHOLD})!")
            sys.exit(1) # Exits with a non-zero code to break the GitHub Actions runner
        else:
            print("Success! Performance threshold verification passed.")
        # 4. Log the trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model_artifact",
            registered_model_name=f"HeartDisease_{model_type}"
        )


if __name__ == "__main__":
    run_training_pipeline()

