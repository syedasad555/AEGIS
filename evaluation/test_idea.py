import json
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

# Test on Fold 1 and Fold 2 baseline errors
with open("evaluation/baseline_cv_results.json") as f:
    base_res = json.load(f)

print("Baseline Acc:", base_res["overall"]["accuracy"], "MacroF1:", base_res["overall"]["macro_f1"])

