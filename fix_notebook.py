import json

# Load notebook
with open('d:/Working/DataMining/DataMiningFinal/Advanced_University_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Before: {len(nb['cells'])} cells")

# Find cell 46 and remove it (index based on our search)
del nb['cells'][46]
print("Removed cell 46")

# Save
with open('d:/Working/DataMining/DataMiningFinal/Advanced_University_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"After: {len(nb['cells'])} cells")
print("Done!")
