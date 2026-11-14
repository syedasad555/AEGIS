import csv

with open('evaluation/ucf_annotations.csv', 'r', encoding='utf-8') as f:
    events = list(csv.DictReader(f))

DEV_VIDEOS = [
    '2018-6_70.mov', '2082-3_70.mov', '341-46_l.mov', '620-72_l.mov',
    '637-147_l.mov', '7399-1_70.mov', '9019-13_l.mov', '913-35_l.mov',
    '3687-18_70.mov', '821-10_l.mov', '9-19_l.mov', '913-37_l.mov'
]
dev_events = [e for e in events if e['video_name'] in DEV_VIDEOS]

folds = {
    'Fold 1': ['2018-6_70.mov', '620-72_l.mov', '9019-13_l.mov', '3687-18_70.mov'],
    'Fold 2': ['2082-3_70.mov', '637-147_l.mov', '913-35_l.mov', '821-10_l.mov'],
    'Fold 3': ['341-46_l.mov', '7399-1_70.mov', '9-19_l.mov', '913-37_l.mov']
}

for fname, fvids in folds.items():
    fev = [e for e in dev_events if e['video_name'] in fvids]
    classes = [e['ground_truth_class'] for e in fev]
    print(f"{fname}: {len(fev)} events -> NORM={classes.count('NORMAL')}, WD={classes.count('WRONG_DIRECTION')}, PS={classes.count('PROLONGED_STOPPING')}")
    print(f"  Videos: {fvids}")

