from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
from pathlib import Path

import cv2
import yaml
from confluent_kafka import Producer

from common.schemas import frame_message


def encode_frame(frame) -> str:
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError('Failed to encode frame')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def run_stream(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    kafka_cfg = config['kafka']
    producer_cfg = config['producer']

    producer = Producer({'bootstrap.servers': kafka_cfg['bootstrap_servers']})
    topic = kafka_cfg['raw_topic']

    source = producer_cfg['video_source']
    source_value = 0 if source == 'webcam' else source
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {source}')

    target_fps = float(producer_cfg.get('target_fps', 20.0))
    sleep_s = max(0.0, (1.0 / target_fps))
    camera_id = producer_cfg.get('camera_id', 'front_center')

    print(f'Streaming frames to Kafka topic={topic} at target_fps={target_fps}')

    try:
        while True:
            start = time.time()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_id = str(uuid.uuid4())
            frame_ts_ms = int(time.time() * 1000)
            payload = frame_message(
                frame_id=frame_id,
                camera_id=camera_id,
                frame_ts_ms=frame_ts_ms,
                source='video_stream',
                image_b64=encode_frame(frame),
            )
            producer.produce(topic, key=frame_id, value=json.dumps(payload).encode('utf-8'))
            producer.poll(0)

            elapsed = time.time() - start
            time.sleep(max(0.0, sleep_s - elapsed))
    except KeyboardInterrupt:
        print('Stopping producer...')
    finally:
        cap.release()
        producer.flush(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run_stream(args.config)


if __name__ == '__main__':
    main()
