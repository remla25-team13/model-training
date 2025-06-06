# Dockerfile
FROM python:3.12.9-slim
WORKDIR /root

RUN apt-get update && apt-get install -y git \ 
        && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY a1_RestaurantReviews_HistoricDump.tsv /root/
COPY training/model_analysis.py /root/
ENTRYPOINT ["python"]
CMD ["model_analysis.py"]