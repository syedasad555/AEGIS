import csv
import json
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path("evaluation")
ann_file = EVAL_DIR / "ucf_annotations.csv"

TEST_VIDEOS = {'688-10_l.mov', '861-13_70.mov', '879-38_l.mov', '906-17_l.mov', '913-36_l.mov'}

events = []
with open(ann_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        events.append(row)

test_events = [e for e in events if e["video_name"] in TEST_VIDEOS]

print(f"=== {len(test_events)} HELD-OUT TEST EVENTS ===")
for i, e in enumerate(test_events, 1):
    print(f"[{i}] Video: {e['video_name']}, TrackID: {e['pedestrian_id']}, GT: {e['ground_truth_class']}, ExpDir: {e['expected_direction']}, Time: {e['start_timestamp']}-{e['end_timestamp']}")
    print(f"    Notes: {e['notes']}")

