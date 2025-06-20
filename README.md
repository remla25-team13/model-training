# model-training

![Build](https://github.com/remla25-team13/model-training/actions/workflows/quality.yml/badge.svg)
![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-70%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-69%25-yellow?logo=pytest![coverage](https://img.shields.io/badge/Coverage-86%25-green?logo=pytest![coverage](https://img.shields.io/badge/Coverage-86%25-green?logo=pytest![coverage](https://img.shields.io/badge/Coverage-unknown-lightgrey)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)
<!--still need to be done dynamically!-->
<!-- ![Flake8](https://img.shields.io/badge/code%20style-flake8-blue)
![Bandit](https://img.shields.io/badge/security-bandit-yellow) -->
![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-9.55-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-7.29-yellow?logo=python![pylint](https://img.shields.io/badge/PyLint-7.34-yellow?logo=python![pylint](https://img.shields.io/badge/PyLint-7.34-yellow?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-blue?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-NA-lightgrey?logo=python&logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)

![ml-score](https://img.shields.io/badge/ML%20Test%20Score-Loading-gray)


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

## 🔒 Linters Used

- **Pylint** with a custom plugin detecting ML-specific code smells 
- **Flake8** with a non-default configuration: increased line length, common ignore rules
- **Bandit** with a tailored `bandit.yaml` to focus on relevant Python security risks

All linters are run automatically in the GitHub Actions workflow.


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
