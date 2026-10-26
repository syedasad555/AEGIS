with open('tools/evaluate_adaptive_inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """        if len(gt_boxes) > 0 and len(valid_pred_boxes_for_track) > 0:
            dists = mm.distances.iou_matrix(gt_boxes, valid_pred_boxes_for_track, max_iou=0.50)"""

replacement1 = """        if len(gt_boxes) > 0 and len(valid_pred_boxes_for_track) > 0:
            # motmetrics.distances.iou_matrix expects [x, y, w, h] format
            gt_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in gt_boxes]
            pred_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in valid_pred_boxes_for_track]
            dists = mm.distances.iou_matrix(gt_boxes_xywh, pred_boxes_xywh, max_iou=0.50)"""

target2 = """        {
            "name": "MOT17-02",
            "video": mot_dir / "MOT17-02" / "mot17-02.mp4",
            "gt": mot_dir / "MOT17-02" / "gt_parsed.json",
            "corridor_roi": (0.15, 0.15, 0.85, 0.65)
        },"""

replacement2 = """        {
            "name": "MOT17-02",
            "video": mot_dir / "MOT17-02" / "mot17-02.mp4",
            "gt": mot_dir / "MOT17-02" / "gt_parsed.json",
            "corridor_roi": (0.20, 0.35, 0.65, 0.60)
        },"""

target3 = """    parser.add_argument("--output", type=str, default="outputs/reports/phase22_adaptive_benchmark.json", help="Output path")"""

replacement3 = """    parser.add_argument("--output", type=str, default="outputs/reports/phase23_adaptive_benchmark.json", help="Output path")"""

assert target1 in content, "target1 not found"
assert target2 in content, "target2 not found"
assert target3 in content, "target3 not found"

content = content.replace(target1, replacement1, 1)
content = content.replace(target2, replacement2, 1)
content = content.replace(target3, replacement3, 1)

with open('tools/evaluate_adaptive_inference.py', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("Successfully updated tools/evaluate_adaptive_inference.py")
