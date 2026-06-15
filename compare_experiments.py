import mlflow

def identify_best_run(experiment_name: str, primary_metric: str):
    """Queries MLflow tracking system to isolate and present the top model run."""
    # Find the target experiment block
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"No experiment tracking records found matching name: {experiment_name}")
        return

    # Programmatically fetch and sort dataframes based on your primary metric
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{primary_metric} DESC"]
    )

    if runs_df.empty:
        print("No recorded training runs found.")
        return

    # Extract the absolute best run row
    best_run = runs_df.iloc[0]
    
    print("=" * 60)
    print("                BEST RUN PERFORMANCE SUMMARY            ")
    print("=" * 60)
    print(f"Run ID:        {best_run['run_id']}")
    print(f"Model Type:    {best_run.get('params.model_type', 'N/A')}")
    print(f"Data Version:  {best_run.get('params.dvc_data_version', 'N/A')}")
    print("-" * 60)
    print(f"PRIMARY METRIC ({primary_metric}): {best_run[f'metrics.{primary_metric}']:.4f}")
    print(f"Accuracy:      {best_run['metrics.accuracy']:.4f}")
    print(f"F1-Score:      {best_run['metrics.f1_score']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    # Query your run database using ROC-AUC as the primary metric
    identify_best_run("Heart_Disease_Prediction_Tracking", "roc_auc_score")
