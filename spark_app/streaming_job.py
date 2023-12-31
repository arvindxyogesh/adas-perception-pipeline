from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import requests
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import LongType, StringType, StructField, StructType


def create_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config('spark.sql.shuffle.partitions', '2')
        .getOrCreate()
    )


def post_with_retries(url: str, payload: Dict[str, Any], timeout_s: int, retries: int) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))


def process_batch(df, batch_id: int, cfg: dict, spark: SparkSession) -> None:
    if df.isEmpty():
        return

    rows = df.select('payload_json').collect()
    output = []
    inference_url = cfg['inference']['url'].rstrip('/') + '/infer_batch'
    batch_size = int(cfg.get('spark', {}).get('inference_batch_size', 8))
    timeout_s = int(cfg.get('inference', {}).get('timeout_s', 8))
    retries = int(cfg.get('inference', {}).get('retries', 2))

    payloads: List[Dict[str, Any]] = [json.loads(row['payload_json']) for row in rows]

    for i in range(0, len(payloads), batch_size):
        chunk = payloads[i:i + batch_size]
        try:
            resp_payload = post_with_retries(
                inference_url,
                {'frames': chunk},
                timeout_s=timeout_s,
                retries=retries,
            )
            for result in resp_payload.get('results', []):
                now_ms = int(time.time() * 1000)
                result['pipeline_latency_ms'] = float(now_ms - int(result['frame_ts_ms']))
                output.append((result['frame_id'], json.dumps(result)))
        except Exception as exc:
            frame_ids = [x.get('frame_id', 'na') for x in chunk]
            print(f'Inference batch failure frame_ids={frame_ids}: {exc}')

    if not output:
        return

    out_df = spark.createDataFrame(output, ['key', 'value'])
    out_df.write.format('kafka').option('kafka.bootstrap.servers', cfg['kafka']['bootstrap_servers']).option(
        'topic', cfg['kafka']['detections_topic']
    ).save()

    print(f'Processed batch={batch_id} frames={len(output)}')


def main() -> None:
    cfg = yaml.safe_load(open('/app/config/settings.yaml', 'r', encoding='utf-8'))
    spark = create_spark('adas-perception-stream')

    schema = StructType(
        [
            StructField('frame_id', StringType(), False),
            StructField('camera_id', StringType(), False),
            StructField('frame_ts_ms', LongType(), False),
            StructField('source', StringType(), True),
            StructField('image_b64', StringType(), False),
        ]
    )

    raw_df = (
        spark.readStream.format('kafka')
        .option('kafka.bootstrap.servers', cfg['kafka']['bootstrap_servers'])
        .option('subscribe', cfg['kafka']['raw_topic'])
        .option('startingOffsets', 'latest')
        .load()
    )

    parsed = raw_df.selectExpr('CAST(value AS STRING) as payload_json').select(
        from_json(col('payload_json'), schema).alias('x'), col('payload_json')
    ).where(col('x').isNotNull())

    query = (
        parsed.writeStream.foreachBatch(lambda d, b: process_batch(d, b, cfg, spark))
        .option('checkpointLocation', '/tmp/spark-checkpoints/adas')
        .start()
    )
    query.awaitTermination()


if __name__ == '__main__':
    main()
