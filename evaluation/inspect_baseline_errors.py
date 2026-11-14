import json

with open("evaluation/baseline_cv_results.json") as f:
    d = json.load(f)

# Let's inspect where baseline errors occurred in dev videos:
# We know confusion matrix is:
# NORMAL: TP=11, FP=6, FN=1 -> 1 NORMAL predicted as STOPPING
# WRONG_DIR: TP=7, FP=0, FN=4 -> 2 predicted as NORMAL, 2 predicted as STOPPING
# STOPPING: TP=9, FP=3, FN=4 -> 4 predicted as NORMAL
print("Baseline per class:")
for k, v in d['overall']['per_class'].items():
    print(k, v)

