#!/bin/bash

python -m review_rating.modeling prepare a1_RestaurantReviews_HistoricDump.tsv -o . -t 0.2
python -m review_rating.modeling train --input-dir . --output-model model.pkl
python -m review_rating.modeling evaluate --input-dir . --model-path model.pkl
