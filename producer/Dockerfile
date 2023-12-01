FROM python:3.10-slim

WORKDIR /app
COPY producer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY producer /app/producer
COPY common /app/common
COPY config /app/config

ENV PYTHONPATH=/app
CMD ["python", "producer/producer.py", "--config", "config/settings.yaml"]
