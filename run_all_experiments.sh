#!/bin/bash

# Ensure we exit immediately if a training command crashes
set -e

echo "========================================================="
echo " Starting Automated MLflow Experimentation Suite"
echo "========================================================="

# Helper function to modify values inside configs/config.yaml safely on a Mac
update_config_val() {
    local key=$1
    local value=$2
    # Mac sed requires the empty quotes '' string for in-place edits without backups
    sed -i '' "s/  $key:.*/  $key: $value/g" configs/config.yaml
}

# --- EXPERIMENT 1: Baseline AdaBoost ---
echo -e "\n[Experiment 1/5] Launching Baseline AdaBoost..."
update_config_val "model_type" "\"AdaBoost\""
update_config_val "n_estimators" "50"
update_config_val "learning_rate" "1.0"
PYTHONPATH=. python -m src.train

# --- EXPERIMENT 2: High Estimator AdaBoost ---
echo -e "\n[Experiment 2/5] Launching Deep AdaBoost Chain..."
update_config_val "model_type" "\"AdaBoost\""
update_config_val "n_estimators" "150"
update_config_val "learning_rate" "0.5"
PYTHONPATH=. python -m src.train

# --- EXPERIMENT 3: Slow Learning AdaBoost ---
echo -e "\n[Experiment 3/5] Launching Conservative AdaBoost..."
update_config_val "model_type" "\"AdaBoost\""
update_config_val "n_estimators" "200"
update_config_val "learning_rate" "0.1"
PYTHONPATH=. python -m src.train

# --- EXPERIMENT 4: Baseline Random Forest ---
echo -e "\n[Experiment 4/5] Launching Shallow Random Forest..."
update_config_val "model_type" "\"RandomForest\""
update_config_val "n_estimators" "50"
update_config_val "learning_rate" "1.0" # Ignored by RF logic but kept for clean logs
PYTHONPATH=. python -m src.train

# --- EXPERIMENT 5: Heavy Tree Ensemble ---
echo -e "\n[Experiment 5/5] Launching Dense Random Forest..."
update_config_val "model_type" "\"RandomForest\""
update_config_val "n_estimators" "150"
update_config_val "learning_rate" "1.0"
PYTHONPATH=. python -m src.train

echo -e "\n========================================================="
echo " All 5 Experiments logged to MLflow successfully!"
echo "========================================================="

# Programmatically determine the winning run
python compare_experiments.py
