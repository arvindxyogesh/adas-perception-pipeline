FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY inference_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY inference_service /app/inference_service
COPY common /app/common

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "inference_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
