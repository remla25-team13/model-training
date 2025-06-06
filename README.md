# model-training

![Build](https://github.com/remla25-team13/model-training/actions/workflows/quality.yml/badge.svg)
![coverage](https://img.shields.io/badge/Coverage-86%25-green?logo=pytest![coverage](https://img.shields.io/badge/Coverage-unknown-lightgrey)logoColor=white)

<!--still need to be done dynamically!-->
<!-- ![Flake8](https://img.shields.io/badge/code%20style-flake8-blue)
![Bandit](https://img.shields.io/badge/security-bandit-yellow) -->

![pylint](https://img.shields.io/badge/PyLint-7.34-yellow?logo=python![pylint](https://img.shields.io/badge/PyLint-7.34-yellow?logo=python![pylint](https://img.shields.io/badge/PyLint-7.34-yellow?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-blue?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-NA-lightgrey?logo=python&logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)logoColor=white)



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

| Tool     | Purpose                            | Configuration        |
|----------|------------------------------------|-----------------------|
| `pylint` | Logic issues, structure, ML rules  | `.pylintrc`, plugin   |
| `flake8` | PEP-8 compliance                   | `.flake8`             |
| `bandit` | Security vulnerability scanning    | `bandit.yaml`         |

## 🔍 Custom Pylint Rules

This project includes custom Pylint rules defined in `pylint_plugins/ml_checks.py` to catch common ML code issues:

- `fit-missing-y`: Warns when `.fit(X)` is called with only one argument — possible missing labels.
- `predict-on-training-data`: Warns when `.predict(X_train)` is used — this may indicate you're evaluating on training data instead of a test split.

These rules are automatically enforced in CI.

## Related Repositories
- [lib-ml](https://github.com/remla25-team13/lib-ml)
- [model-service](https://github.com/remla25-team13/model-service)
- [operation](https://github.com/remla25-team13/operation)