FROM apache/spark:3.5.1

USER root
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
COPY spark_app/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY spark_app /app/spark_app
COPY config /app/config

ENV PYTHONPATH=/app
USER 185
