# Dockerfile
FROM python:3.12-slim

WORKDIR /root

RUN apt-get update && apt-get install --no-install-recommends -y git \ 
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /root/
RUN pip install -r requirements.txt

COPY run_all.sh /root/
RUN chmod +x /root/run_all.sh

COPY data/raw/a1_RestaurantReviews_HistoricDump.tsv /root/

# Copy the `review_rating` directory
COPY review_rating /root/review_rating

CMD ["/root/run_all.sh"]
