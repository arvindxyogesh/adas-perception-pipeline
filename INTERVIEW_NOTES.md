# Interview Notes: ADAS Streaming Perception

## What This Demonstrates
- Real-time stream ingestion (Kafka)
- Stateful micro-batch processing (Spark Structured Streaming)
- GPU inference service design (FastAPI + CUDA)
- Computer vision stack for ADAS primitives (objects + lane lines)
- Production reliability concerns (healthchecks, retries, metrics)

## Key Engineering Decisions
1. Decoupled producer, stream processor, and inference service for independent scaling.
2. Used Kafka topics `raw_frames` and `detections` to isolate ingestion from downstream analytics.
3. Added batch inference endpoint to reduce per-request overhead and improve GPU utilization.
4. Added request retries with bounded backoff to handle transient inference outages.
5. Exposed Prometheus metrics for service-level observability.

## Metrics to Present
- FPS (target ~20)
- End-to-end latency p50/p95 (target p95 < 150 ms in simulated setup)
- Inference latency (per frame)
- Throughput (frames/sec)
- Precision/Recall from offline evaluation script

## What You Would Build Next in Production
1. Replace Torch fallback with real TensorRT engine loading path.
2. Add schema registry and contract tests.
3. Add CI for integration tests with mocked Kafka and inference.
4. Add autoscaling policy based on lag + inference latency.
5. Add drift and scene-shift monitoring for model quality in the field.
