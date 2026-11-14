import json

with open('evaluation/final_held_out_comparison.json') as f:
    d = json.load(f)

print('=== BASELINE HELD-OUT METRICS ===')
print('Accuracy:', d['baseline']['metrics']['accuracy'])
print('Macro F1:', d['baseline']['metrics']['macro_f1'])
print('Confusion matrix (NORM, WD, PS):', d['baseline']['metrics']['confusion_matrix'])
print('Per-class:')
for k, v in d['baseline']['metrics']['per_class'].items():
    print(' ', k, v)

print('\n=== IMPROVED HELD-OUT METRICS ===')
print('Accuracy:', d['improved']['metrics']['accuracy'])
print('Macro F1:', d['improved']['metrics']['macro_f1'])
print('Confusion matrix (NORM, WD, PS):', d['improved']['metrics']['confusion_matrix'])
print('Per-class:')
for k, v in d['improved']['metrics']['per_class'].items():
    print(' ', k, v)

print('\n=== EVENT-BY-EVENT COMPARISON ===')
b_events = { (e['video_name'], e['track_id']): e for e in d['baseline']['events'] }
i_events = { (e['video_name'], e['track_id']): e for e in d['improved']['events'] }

for k in sorted(b_events.keys()):
    be = b_events[k]
    ie = i_events[k]
    vname, tid = k
    print(f"{vname:15s} ID:{tid:4d} | GT: {be['ground_truth']:18s} | Base: {be['predicted']:18s} | Imp: {ie['predicted']:18s} | BaseOk: {be['correct']} | ImpOk: {ie['correct']}")

