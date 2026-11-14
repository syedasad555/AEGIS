import csv

with open('evaluation/ucf_annotations.csv', 'r', encoding='utf-8') as f:
    events = list(csv.DictReader(f))

DEV_VIDEOS = [
    '2018-6_70.mov', '2082-3_70.mov', '341-46_l.mov', '620-72_l.mov',
    '637-147_l.mov', '7399-1_70.mov', '9019-13_l.mov', '913-35_l.mov',
    '3687-18_70.mov', '821-10_l.mov', '9-19_l.mov', '913-37_l.mov'
]
dev_events = [e for e in events if e['video_name'] in DEV_VIDEOS]

for e in dev_events:
    print(f"{e['video_name']:16s} ID:{e['pedestrian_id']:>4s} GT:{e['ground_truth_class']:<19s} ExpDir:{e['expected_direction']:<6s} Notes:{e['notes']}")

