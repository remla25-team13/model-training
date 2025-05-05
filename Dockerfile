# Dockerfile
FROM python:3.12.9-slim
WORKDIR /root
COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY a1_RestaurantReviews_HistoricDump.tsv /root/
COPY model_analysis.py /root/
ENTRYPOINT ["python"]
CMD ["model_analysis.py"]