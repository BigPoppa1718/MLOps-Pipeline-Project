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
