# Heart Disease Prediction Pipeline

An enterprise-ready, modular machine learning pipeline built with Scikit-Learn to preprocess clinical data, handle missing values safely, and train an AdaBoost binary classification model to predict heart disease presence.

## Project Architecture
```text
├── .github/workflows/   # CI/CD orchestration scripts
│   └── ci.yml
├── configs/             # Hyperparameters and path tracking maps
│   └── config.yaml
├── metrics/             # Local metrics tracking exports (Git ignored)
├── model/               # Serialized binary model storage (Git ignored)
├── src/                 # Codebase logic partition modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   └── train.py
├── tests/               # Automation testing suite
│   ├── 
│   └── test_pipeline.py
├── .gitignore           # File tracking isolation exclusion patterns
├── README.md            # Execution guidance and setup manual
└── requirements.txt     # Pinned dependency environment blueprint
```

## Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Installation & Environment Setup
Clone this repository to your local computer, create a virtual environment, and install the pinned requirements:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Mac/Linux)
source venv/bin/activate

# Activate the environment (Windows)
venv\Scripts\activate

# Install exact pinned versions
pip install -r requirements.txt
```

## How to Run

### Execute Training
Run the core training controller from the project root. This module imports parameters from `configs/config.yaml`, generates the data, runs the transformation transformer streams, builds the pipeline, and saves your assets:

```bash
python src/train.py
```
* **Model output location:** `model/model.pkl`
* **Performance metrics output location:** `metrics/metrics.json`

### Execute Automation Testing
Run the comprehensive verification check sequence to test module structures and synthetic data logic definitions:

```bash
pytest
```

## Continuous Integration
This repository is configured with a GitHub Actions workflow (`.github/workflows/ci.yml`). Every time you push code or open a Pull Request targeting the `main` branch, the system will automatically spin up a test container, load your environment dependencies, and execute `pytest` to confirm code health.

## Data Version Control (DVC)

This project uses DVC to separate raw datasets from source control code history. The dataset asset is managed by DVC and linked to a local mock remote located at `/tmp/dvc_local_remote`.

### Grader Data Sync
To pull the underlying training dataset into your local machine after cloning the repository, run:

```bash
dvc pull
```

This commands reads the metadata from `data/heart_disease.csv.dvc`, contacts the remote path, and extracts the verified `data/heart_disease.csv` file into place automatically.

## 4. CI/CD Automation Pipeline (GitHub Actions)

This repository includes a production-grade, multi-job automation workflow managed via GitHub Actions (`.github/workflows/ci.yml`). The automation strategy guarantees that code modifications never compromise codebase logic or model performance quality baselines.

### Workflow Orchestration Design
The pipeline is divided into two distinct, sequential jobs where execution progression is dependent on quality gates:

```text
  [ Push / Pull Request to main ]
                 │
                 ▼
       ┌───────────────────┐
       │   1. run-tests    │  ◄─── Installs dependencies, builds dataset, 
       └─────────┬─────────┘      and runs all 11 pytest suites.
                 │
                 │ (Passes Cleanly)
                 ▼
       ┌───────────────────┐
       │  2. train-model   │  ◄─── Runs training script, checks performance 
       └───────────────────┘      gate, and safe-skips database logs.
```

### Automation Details & Quality Gates

#### 1. Code Verification & Data Validation Job (`run-tests`)
* **Trigger Conditions:** Fires automatically on any `push` or `pull_request` targeting the `main` branch.
* **Execution Stack:** Environments are provisioned with **Python 3.12** and updated to **Node.js 24** execution layers to prevent runner deprecation blocks.
* **Validation Tasks:** Installs pinned dependencies, builds a synthetic dataset sample on the runner, and fires the complete `pytest` validation suite (`pytest -v`).
* **Gate constraint:** If any of the 6 preprocessing unit tests, 3 data profile validation tests, or 2 model behavioral tests fail, the workflow aborts immediately.

#### 2. Dependent Model Performance Job (`train-model`)
* **Dependencies:** This job will only unlock if the `run-tests` job finishes with a completely successful exit status (`needs: run-tests`).
* **Performance Gate Integration:** The training module (`src/train.py`) enforces a strict **Minimum Performance Gateway**. If the trained model drops below a **65% Accuracy threshold (`MIN_ACCURACY_THRESHOLD = 0.65`)**, the script throws an error and exits with system code `1`, causing the GitHub Actions tab to turn red.
* **CI Environment Isolation:** When executed within a GitHub container, the training logic automatically sets an environment override to safely skip local SQLite (`sqlite:///mlflow.db`) backend lookups. This allows performance threshold validations to complete without dependency crashes.

### Tracking Workflow Actions Run
Graders can verify pipeline execution history by navigating to the **Actions** tab of this repository. A successful run will display a solid **green checkmark**, confirming that all underlying structural tests and model performance constraints passed cleanly.

## MONITORING SECTION
## Data Drift Monitoring

The project includes a monitoring script (`src/monitor_drift.py`) that compares the training dataset against a simulated production dataset using Evidently.

### Drift Monitoring Results

The reference dataset consisted of the original heart disease training data, while the production dataset was generated with intentionally drifted feature distributions to simulate changes that may occur after deployment.

The drift analysis evaluated 11 model features and detected drift in 3 features, resulting in an overall drift share of 27.27%.

### Which Features Showed Drift and Why?

The primary drifted features were:

* **Age** – The production dataset was generated with a higher average patient age distribution.
* **Cholesterol** – Cholesterol values were intentionally shifted upward to simulate a changing patient population.
* **Chest Pain Type** – Category frequencies were modified to increase the proportion of asymptomatic cases.

These modifications were introduced deliberately to simulate realistic changes that could occur as patient demographics and clinical characteristics evolve over time.

### Would This Drift Affect Model Performance?

Potentially yes. Age and cholesterol are important predictive variables in the heart disease model. Significant shifts in these distributions may reduce model accuracy because the production data no longer closely resembles the data used during training.

Changes in chest pain type frequencies may also affect model predictions if the model relies heavily on that categorical feature.

However, the overall drift share of 27.27% remained below the configured monitoring threshold of 30%, indicating that the model is still operating within acceptable limits.

### Recommended Action

At this time, I would recommend **continued monitoring** rather than immediate retraining.

Since the overall drift level remains below the alert threshold, retraining is not yet justified. If drift continues to increase or model performance metrics begin to decline, further investigation and model retraining should be considered.

