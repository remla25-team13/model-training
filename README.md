# model-training

![Build](https://github.com/remla25-team13/model-training/actions/workflows/quality.yml/badge.svg)

<!--still need to be done dynamically!-->
![Flake8](https://img.shields.io/badge/code%20style-flake8-blue)
![Bandit](https://img.shields.io/badge/security-bandit-yellow)

![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-10.00-brightgreen?logo=python![pylint](https://img.shields.io/badge/PyLint-NA-lightgrey?logo=python&logoColor=white)logoColor=white)logoColor=white)



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

## Related Repositories
- [lib-ml](https://github.com/remla25-team13/lib-ml)
- [model-service](https://github.com/remla25-team13/model-service)
- [operation](https://github.com/remla25-team13/operation)