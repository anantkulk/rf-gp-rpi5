
from mp_api.client import MPRester
import random

#Calls API using key from Materials Project to find materials
with MPRester("8lwWlaQJEGBQ5YWwRG9ebji8JKTZ5Cyl") as mpr:
    mats = mpr.materials.summary.search(
        elements=["Na"],
        exclude_elements=[],
        num_elements = (2,3),
        energy_above_hull=(0,0.01),
        fields=[
            "elements",
            "formula_pretty",
            "formation_energy_per_atom",
        ],
        all_fields=False
    )

binary = []
ternary = [ ]
for m in mats: 
    if len(m.elements) == 2:
        binary.append(m)
    else:
        ternary.append(m)

#Only 87 stable binary Sodium intermetallics on MP, Pick 413 random Ternary to get to 500 strucutres
data = random.sample(ternary, 413)
data = data + binary
print(len(mats))
print(len(binary))
print(len(ternary))
print(len(data))

import json

final_data = []
for m in data:
    final_data.append({
        "elements": [str(n) for n in m.elements],
        "formula": m.formula_pretty,
        "formation_energy": m.formation_energy_per_atom
    })
with open("structures_data.txt", "w") as f:
    json.dump(final_data,f)