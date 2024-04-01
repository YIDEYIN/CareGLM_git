import json

with open('cMedQA.json', 'r') as f1:
    lst1 = json.load(f1)
with open('cMedQA2.json', 'r') as f2:
    dict2 = json.load(f2)

lst1.append(dict2)

with open('merged.json', 'w') as f:
    json.dump(lst1, f, indent=4)