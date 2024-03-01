from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def iou(box_a: List[float], box_b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(preds: List[Dict], gts: List[Dict], iou_thr: float = 0.5) -> Tuple[float, float]:
    gt_by_frame = {x['frame_id']: x.get('objects', []) for x in gts}
    tp = fp = fn = 0

    for pred in preds:
        frame_id = pred['frame_id']
        dets = pred.get('detections', [])
        gt_objs = gt_by_frame.get(frame_id, [])
        matched = set()

        for d in dets:
            found = False
            for idx, g in enumerate(gt_objs):
                if idx in matched:
                    continue
                if d['label'] != g['label']:
                    continue
                if iou(d['bbox'], g['bbox']) >= iou_thr:
                    tp += 1
                    matched.add(idx)
                    found = True
                    break
            if not found:
                fp += 1
        fn += max(0, len(gt_objs) - len(matched))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    gts = load_jsonl(args.ground_truth)
    precision, recall = evaluate(preds, gts)
    print(json.dumps({'precision': precision, 'recall': recall}, indent=2))


if __name__ == '__main__':
    main()
