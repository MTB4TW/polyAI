#!/usr/bin/env python
"""Audit generated combat maps against the Polytopia wiki's map generation rules (polytopia.fandom.com/wiki/Map_Generation).

    python tools/audit_map_generation.py pol_env/Tribes/levels/phase2_pool_1v1/train pol_env/Tribes/levels/phase2_pool_1v1/eval_validation

Checks, on Drylands maps:
- the water share;
- capitals in different quadrants (the generator's domains);
- village and capital spacing, and villages away from the edge;
- ruin count and placement;
- resources only within 2 tiles of a city or village;
- each tribe's field/forest/mountain shares (mountain modifier first, then forest);
- resource shares by ring (adjacent = inner, distance 2 = outer).

Tile ownership is approximated by the nearest capital, so shares near the border mix the two tribes.
"""
import glob, sys, collections, statistics as st
TRIBE = {0: "XIN_XI", 1: "IMPERIUS", 2: "BARDUR", 3: "OUMAJI"}
# Expected terrain shares per tribe (wiki: mountain modifier first scales forest+field, then forest scales field).
def expected(t):
    m, f = 14.0, 38.0
    mm = {"XIN_XI": 1.5, "OUMAJI": 0.5}.get(t, 1.0); fm = {"BARDUR": 0.8, "OUMAJI": 0.2}.get(t, 1.0)
    m2 = m * mm; rest = 100 - m2; f2 = f * rest / 86.0 * fm
    return {"field": 100 - m2 - f2, "forest": f2, "mountain": m2}
def cheb(a, b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))
files = sorted(f for d in (sys.argv[1:] or ["pol_env/Tribes/levels/phase2_pool_1v1/train"]) for f in glob.glob(d.rstrip("/") + "/*.csv"))
agg = collections.defaultdict(lambda: collections.Counter())
res_ring = collections.defaultdict(lambda: collections.Counter())
stats = collections.defaultdict(list)
violations = collections.Counter()
for path in files:
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    n = len(rows)
    grid = {(x, y): rows[x][y].split(":") for x in range(n) for y in range(n)}
    caps = {(x, y): TRIBE[int(r)] for (x, y), (t, r) in grid.items() if t == "c"}
    vills = [p for p, (t, r) in grid.items() if t == "v"]
    centers = list(caps) + vills
    water = sum(1 for (t, r) in grid.values() if t in ("s", "d"))
    stats["water_pct"].append(100 * water / (n * n))
    stats["villages"].append(len(vills))
    q = [((x >= n // 2), (y >= n // 2)) for (x, y) in caps]  # generator domains: rows/cols [0, n//2) and [n//2, n)
    stats["capitals_same_quadrant"].append(int(len(set(q)) < len(q)))
    stats["capital_distance"].append(cheb(*list(caps)) if len(caps) == 2 else -1)
    for i, a in enumerate(centers):
        for b in centers[i + 1:]:
            if cheb(a, b) < 3: violations["villages/cities closer than 3 (within 2 tiles)"] += 1
    for (x, y) in vills:
        if min(x, y, n - 1 - x, n - 1 - y) < 2: violations["village within 2 tiles of the edge"] += 1
    ruins = [p for p, (t, r) in grid.items() if r == "r"]
    stats["ruins"].append(len(ruins))
    if len(ruins) != {11: 4, 14: 5, 16: 7, 18: 9, 20: 11, 30: 23}.get(n, len(ruins)): violations["ruin count differs from the wiki table"] += 1
    for p in ruins:
        if any(cheb(p, c) <= 1 for c in centers): violations["ruin next to a village/city"] += 1
        if any(cheb(p, o) <= 1 for o in ruins if o != p): violations["ruin next to a ruin"] += 1
    for p, (t, r) in grid.items():
        if t in ("c", "v"): continue
        owner = min(caps, key=lambda c: (cheb(p, c), c))  # nearest capital = whose land it is (approximation)
        tribe = caps[owner]
        kind = {".": "field", "f": "forest", "m": "mountain"}.get(t)
        if kind: agg[tribe][kind] += 1
        d = min(cheb(p, c) for c in centers)
        ring = "inner" if d == 1 else "outer2" if d == 2 else "beyond"
        if r and r != "r":
            if ring == "beyond": violations["resource more than 2 tiles from any city/village"] += 1
        if kind:
            res = {"f": "fruit", "c": "crop", "a": "animal", "o": "metal"}.get(r, "none")
            res_ring[(tribe, kind, ring)][res] += 1
print(f"maps audited: {len(files)}")
for k in ("water_pct", "villages", "ruins", "capital_distance"):
    v = stats[k]; print(f"  {k}: mean {st.mean(v):.2f}, min {min(v)}, max {max(v)}")
print(f"  maps with both capitals in one quadrant: {sum(stats['capitals_same_quadrant'])}")
print("rule violations:", dict(violations) or "none")
print("\nterrain share of each tribe's land (non-city tiles, nearest-capital split) vs wiki:")
for t, c in sorted(agg.items()):
    tot = sum(c.values()); e = expected(t)
    print(f"  {t:9s} " + "  ".join(f"{k} {100*c[k]/tot:5.1f}% (wiki {e[k]:4.1f})" for k in ("field", "forest", "mountain")) + f"  [{tot} tiles]")
print("\nresources by ring (share of that terrain in that ring) — wiki base: field inner fruit 37.5/crop 37.5, outer 12.5/12.5; forest animal inner 50, outer 15.8; mountain metal inner 78.6, outer 21.4")
for t in sorted({k[0] for k in res_ring}):
    for kind, rs in (("field", ("fruit", "crop")), ("forest", ("animal",)), ("mountain", ("metal",))):
        line = []
        for ring in ("inner", "outer2"):
            c = res_ring[(t, kind, ring)]; tot = sum(c.values())
            if tot: line.append(f"{ring}: " + ", ".join(f"{r} {100*c[r]/tot:4.1f}%" for r in rs) + f" (n={tot})")
        print(f"  {t:9s} {kind:8s} " + " | ".join(line))
