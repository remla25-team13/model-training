# model-training

![coverage](https://img.shields.io/badge/Coverage-75%25-green?logo=pytest![coverage](https://img.shields.io/badge/Coverage-75%25-green?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-71%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-86%25-green?logo=pytest&logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)
![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest![ml-score](https://img.shields.io/badge/ML%20Test%20Score-4%2F4-brightgreen?logo=pytest&logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)


This repository is part of the REMLA25 project by team 13 and contains the machine learning training pipeline for sentiment analysis on restaurant reviews.

## Features
- Loads the historic restaurant reviews dataset.
- Preprocesses data using `lib-ml`.
- Trains and evaluates the sentiment model.
- Versions and releases the model for deployment.

## Usage
Clone the repository and run the training:

```bash
git clone https://github.com/remla25-team13/model-training.git
cd model-training
docker build -t model-training .
docker run --rm model-training
```

Or run the pipeline through DVC:
```bash
dvc repro
```

### Loading the dataset
We load the dataset automatically through `dvc repro`. For this to work properly, you need to define the `GDRIVE_CREDENTIALS_DATA` environment variable. `GDRIVE_CREDENTIALS_DATA` holds the necessary information to load the Google Cloud Service Account responsible for managing our data.

## Project Overview

![cookiecutter](https://img.shields.io/badge/CCDS-template-blue?logo=cookiecutter&logoColor=yellow)
```
model-training/
├── .github/               # CI configuration and workflows
├── data/                  # DVC-managed input/output data (not tracked in Git)
├── metrics/               # Evaluation metrics and ML test score output
├── review_rating/         # Main source code for model, pipeline, and tests
│   ├── __init__.py
│   ├── data/              # Data loading & preprocessing
│   ├── modeling/          # Model training and evaluation
│   ├── tests/             # Unit and integration tests
│   └── utils.py           # Reusable utilities
├── dvc.yaml               # DVC pipeline definition
├── Dockerfile             # Container setup for training environment
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Dev dependencies (linters, test tools)
├── pyproject.toml         # Pylint config and tool integration
└── README.md              # Project documentation
```

## 🔒 Linters Used
![pylint](https://img.shields.io/badge/PyLint-9.84-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-9.84-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python&logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)
![Flake8](https://img.shields.io/badge/code%20style-flake8-blue)
![Bandit](https://img.shields.io/badge/security-bandit-yellow)

- **Pylint** with a custom plugin detecting ML-specific code smells 
- **Flake8** with a non-default configuration: increased line length, common ignore rules
- **Bandit** with a tailored `bandit.yaml` to focus on relevant Python security risks

This project enforces code quality through three linters, all run automatically in the **GitHub Actions workflow** defined in [`quality.yml`](.github/workflows/quality.yml):

### ✅ How They Are Run

- **Pylint**
  - Run with:
    ```bash
    PYTHONPATH=. pylint review_rating --rcfile=.pylintrc
    ```
  - Includes a custom plugin (`pylint_plugins/ml_checks.py`) that detects machine learning-specific code smells.
  - The Pylint score is parsed in CI and used to update the README badge.

- **Flake8**
  - Run with:
    ```bash
    flake8 review_rating
    ```
  - Configured via `requirements-dev.txt` with extended line length and common ignore rules.

- **Bandit**
  - Run with:
    ```bash
    bandit -r review_rating -c bandit.yaml
    ```
  - Uses a tailored config to detect security issues relevant to ML workflows.

## 🤖 Custom Pylint Rules for ML Code Smells

This project includes a custom Pylint plugin (`pylint_plugins/ml_checks.py`) that also implements a ML-specific code smell detector inspired by [ml-smells](https://hynn01.github.io/ml-smells/). These rules help prevent common mistakes in data science and machine learning workflows:

| Rule ID  | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `W9001`  | `predict()` called on training data (`X_train`) — may indicate data leakage|
| `W9002`  | `.values` used on DataFrames — prefer `df.to_numpy()`                      |

These checks are integrated into our CI using `pylint` with a custom `.pylintrc` and are automatically run and scored.

## 🧪 ML Test Score

The ML Test Score is automatically computed in the GitHub workflow and follows the [ML Test Score methodology](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/).

| Category             | Status  |
|----------------------|---------|
| Feature & Data       | ✅      |
| Model Development    | ✅      |
| ML Infrastructure    | ✅      |
| Monitoring           | ✅      |


## Related Repositories
- [lib-ml](https://github.com/remla25-team13/lib-ml)
- [model-service](https://github.com/remla25-team13/model-service)
- [operation](https://github.com/remla25-team13/operation)
