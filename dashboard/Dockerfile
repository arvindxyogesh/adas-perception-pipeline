FROM python:3.10-slim

WORKDIR /app
COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY dashboard /app/dashboard
COPY common /app/common
COPY config /app/config

ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
