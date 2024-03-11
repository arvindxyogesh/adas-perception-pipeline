from __future__ import annotations

import json
import time
from collections import deque

import pandas as pd
import plotly.express as px
import streamlit as st
from confluent_kafka import Consumer

from common.metrics import percentile

st.set_page_config(page_title='ADAS Streaming Dashboard', layout='wide')
st.title('Real-Time ADAS Perception Metrics')

bootstrap_servers = st.sidebar.text_input('Kafka Bootstrap Servers', 'kafka:9092')
topic = st.sidebar.text_input('Detections Topic', 'detections')

if 'consumer' not in st.session_state:
    st.session_state.consumer = Consumer(
        {
            'bootstrap.servers': bootstrap_servers,
            'group.id': 'adas-dashboard',
            'auto.offset.reset': 'latest',
        }
    )
    st.session_state.consumer.subscribe([topic])

if 'records' not in st.session_state:
    st.session_state.records = deque(maxlen=3000)

placeholder = st.empty()

while True:
    msg = st.session_state.consumer.poll(0.2)
    if msg is not None and not msg.error():
        payload = json.loads(msg.value().decode('utf-8'))
        st.session_state.records.append(payload)

    data = list(st.session_state.records)
    if data:
        latencies = [float(x.get('pipeline_latency_ms', 0.0)) for x in data]
        inference = [float(x.get('inference_ms', 0.0)) for x in data]
        frame_ids = [x['frame_id'] for x in data[-80:]]

        fps = 0.0
        if len(data) > 1:
            dt = (data[-1]['processing_ts_ms'] - data[0]['processing_ts_ms']) / 1000.0
            fps = len(data) / dt if dt > 0 else 0.0

        p50 = sum(latencies) / len(latencies)
        p95 = percentile(latencies, 0.95)

        with placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('FPS (window)', f'{fps:.2f}')
            c2.metric('Latency p50 (ms)', f'{p50:.1f}')
            c3.metric('Latency p95 (ms)', f'{p95:.1f}')
            c4.metric('Inference avg (ms)', f'{sum(inference)/len(inference):.1f}')

            df = pd.DataFrame(
                {
                    'idx': list(range(len(data[-80:]))),
                    'latency_ms': latencies[-80:],
                    'inference_ms': inference[-80:],
                    'frame_id': frame_ids,
                }
            )
            st.plotly_chart(px.line(df, x='idx', y=['latency_ms', 'inference_ms'], title='Recent Latency Trend'), use_container_width=True)

            det_count = [len(x.get('detections', [])) for x in data[-80:]]
            lanes_count = [len(x.get('lanes', [])) for x in data[-80:]]
            df_counts = pd.DataFrame({'idx': list(range(len(det_count))), 'detections': det_count, 'lanes': lanes_count})
            st.plotly_chart(px.bar(df_counts, x='idx', y=['detections', 'lanes'], barmode='group', title='Detections and Lanes per Frame'), use_container_width=True)

    time.sleep(0.3)
