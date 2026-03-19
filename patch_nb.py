import json

with open('mnist.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if any("X = X / 255.0" in line for line in source):
            cell['source'] = [
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(X)\n",
                "X = pd.DataFrame(X_scaled, columns=X.columns)"
            ]
            changed = True

if changed:
    with open('mnist.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated.")
else:
    print("Pattern not found.")
