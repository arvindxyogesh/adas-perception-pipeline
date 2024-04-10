from scripts.evaluate_precision_recall import evaluate


def test_precision_recall_simple_case() -> None:
    preds = [
        {
            'frame_id': 'f1',
            'detections': [
                {'label': 'car', 'bbox': [0, 0, 10, 10]},
                {'label': 'pedestrian', 'bbox': [20, 20, 30, 30]},
            ],
        }
    ]
    gts = [
        {
            'frame_id': 'f1',
            'objects': [
                {'label': 'car', 'bbox': [0, 0, 10, 10]},
                {'label': 'pedestrian', 'bbox': [21, 21, 31, 31]},
            ],
        }
    ]

    precision, recall = evaluate(preds, gts, iou_thr=0.5)
    assert precision == 1.0
    assert recall == 1.0
