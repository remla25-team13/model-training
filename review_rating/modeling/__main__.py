"""Main entry point for training and evaluating a sentiment analysis model on restaurant reviews."""

import argparse
from pathlib import Path

from dvclive import Live
from review_rating.modeling.evaluate import evaluate_model
from review_rating.modeling.prepare_data import prepare_data
from review_rating.modeling.train import train_model

DEFAULT_DATA_PATH = "data/a1_RestaurantReviews_HistoricDump.tsv"


parser = argparse.ArgumentParser(
    description="Train and evaluate a sentiment analysis model on restaurant reviews."
)
subparsers = parser.add_subparsers(dest="command", required=True)

# Common arguments
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument(
    "--random-state",
    "-r",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)

# Data preparation command
prepare_parser = subparsers.add_parser(
    "prepare",
    help="Prepare and split the data.",
    parents=[common_parser],
)
prepare_parser.add_argument(
    "data_path",
    type=str,
    default=DEFAULT_DATA_PATH,
    help=f"Path to the TSV file (default: {DEFAULT_DATA_PATH})",
    nargs="?",
)
prepare_parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="processed_data",
    help="Directory to save the processed data splits.",
)
prepare_parser.add_argument(
    "--test-split",
    "-t",
    type=float,
    default=0.2,
    help="Proportion of the dataset to include in the test split.",
)

# Training command
train_parser = subparsers.add_parser(
    "train",
    help="Train the model.",
    parents=[common_parser],
)
train_parser.add_argument(
    "--input-dir",
    "-i",
    type=str,
    default="processed_data",
    help="Directory containing the processed data splits.",
)
train_parser.add_argument(
    "--output-model",
    "-o",
    type=str,
    default="sentiment_model",
    help="Path to save the trained model.",
)

# Evaluation command
eval_parser = subparsers.add_parser(
    "evaluate",
    help="Evaluate the model.",
    parents=[common_parser],
)
eval_parser.add_argument(
    "--input-dir",
    "-i",
    type=str,
    default="processed_data",
    help="Directory containing the processed data splits.",
)
eval_parser.add_argument(
    "--model-path",
    "-m",
    type=str,
    default="sentiment_model",
    help="Path to the trained model.",
)
eval_parser.add_argument(
    "--metrics-path",
    "-o",
    type=str,
    default="results",
    help="File to save evaluation results.",
)

args = parser.parse_args()

if args.command == "prepare":
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    prepare_data(
        input_path=args.data_path,
        output_dir=args.output_dir,
        test_split=args.test_split,
        random_state=args.random_state,
    )
    print(f"Data prepared and saved to {args.output_dir}")

elif args.command == "train":
    model_dir = Path(args.output_model).parent
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)

    with Live() as live_logger:
        train_model(
            input_dir=args.input_dir,
            output_path=args.output_model,
            live_logger=live_logger,
        )
    print(f"Model trained and saved to {args.output_model}")

elif args.command == "evaluate":
    evaluate_model(
        input_dir=args.input_dir,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )
