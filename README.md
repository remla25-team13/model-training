# model-training

![Build](https://github.com/remla25-team13/model_training/actions/workflows/ci.yml/badge.svg)


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

## Related Repositories
- [lib-ml](https://github.com/remla25-team13/lib-ml)
- [model-service](https://github.com/remla25-team13/model-service)
- [operation](https://github.com/remla25-team13/operation)
