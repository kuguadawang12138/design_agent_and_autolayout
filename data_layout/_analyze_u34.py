# -*- coding: utf-8 -*-
import json
from collections import defaultdict

p = r"d:\260805_pcbdesign\data_layout\U34_Comps.json"
d = json.load(open(p, encoding="utf-8"))
comps = d["components"]
print("n", len(comps), "top_keys", list(d.keys()))

by = {}
for c in comps:
    bb = c["bBox"]
    w = bb["x2"] - bb["x1"]
    h = bb["y2"] - bb["y1"]
    by[c["refName"]] = c
    nets = sorted({pin["pinNetName"] for pin in c["pins"]})
    print(
        f"{c['refName']:6} rot={float(c['rotation']):6.1f} {c['Layer']:6} "
        f"o=({c['origin']['x']:.1f},{c['origin']['y']:.1f}) "
        f"bbox={w:.1f}x{h:.1f} pins={len(c['pins'])} nets={nets}"
    )

u = by["U34"]
# pin sides relative to origin
ox, oy = u["origin"]["x"], u["origin"]["y"]
sides = defaultdict(list)
for pin in u["pins"]:
    dx = pin["pin-xy"]["x"] - ox
    dy = pin["pin-xy"]["y"] - oy
    if abs(dx) >= abs(dy):
        side = "R" if dx > 0 else "L"
    else:
        side = "T" if dy > 0 else "B"
    sides[side].append((pin["pinNumber"], pin["pinNetName"], pin["pin-xy"]["x"], pin["pin-xy"]["y"]))

print("\n=== U34 pins by side ===")
for s in "R L T B":
    print(s, len(sides[s]))
    for item in sides[s]:
        print(" ", item)

# nets: which comps share
net_refs = defaultdict(set)
for c in comps:
    for pin in c["pins"]:
        net_refs[pin["pinNetName"]].add(c["refName"])

print("\n=== nets touching U34 ===")
u_nets = {pin["pinNetName"] for pin in u["pins"]}
for net in sorted(u_nets):
    refs = sorted(net_refs[net])
    print(f"{net:24} {refs}")

print("\n=== nets NOT touching U34 ===")
for net, refs in sorted(net_refs.items()):
    if net not in u_nets:
        print(f"{net:24} {sorted(refs)}")
