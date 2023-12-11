#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP=${1:-localhost:9092}

docker exec -it $(docker ps --filter name=kafka --format '{{.ID}}' | head -n 1) \
  kafka-topics.sh --create --if-not-exists --topic raw_frames --bootstrap-server "$BOOTSTRAP" --partitions 3 --replication-factor 1

docker exec -it $(docker ps --filter name=kafka --format '{{.ID}}' | head -n 1) \
  kafka-topics.sh --create --if-not-exists --topic detections --bootstrap-server "$BOOTSTRAP" --partitions 3 --replication-factor 1

echo "Topics ensured: raw_frames, detections"
