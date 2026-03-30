# Verify no overlapping stopes in optimizer output
# Uses outputs/stopes_results.csv (columns: IX0, IY0, IZ0 = block indices)
import pandas as pd
import os

# Stope size in blocks (same as stope_optimizer: 20m x 5m x 30m with 5m blocks)
LB, TB, HB = 4, 1, 6  # blocks in X, Y, Z

csv_path = os.path.join(os.path.dirname(__file__), "outputs", "stopes_results.csv")
if not os.path.isfile(csv_path):
    print(f"File not found: {csv_path}")
    print("Run the optimizer first to generate outputs/stopes_results.csv")
    exit(1)

df = pd.read_csv(csv_path)
n_stopes = len(df)

# Expect columns from stope_optimizer: IX0, IY0, IZ0 (block indices)
required = {"IX0", "IY0", "IZ0"}
missing = required - set(df.columns)
if missing:
    print(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
    exit(1)

visited_blocks = set()
overlaps_found = 0
overlap_pairs = []

for idx, row in df.iterrows():
    ix0 = int(row["IX0"])
    iy0 = int(row["IY0"])
    iz0 = int(row["IZ0"])
    rank = row.get("RANK", idx + 1)

    for dx in range(LB):
        for dy in range(TB):
            for dz in range(HB):
                block = (ix0 + dx, iy0 + dy, iz0 + dz)
                if block in visited_blocks:
                    overlaps_found += 1
                    overlap_pairs.append((rank, block))
                visited_blocks.add(block)

blocks_per_stope = LB * TB * HB
print(f"Stopes in CSV: {n_stopes}")
print(f"Blocks per stope: {blocks_per_stope} (X={LB} x Y={TB} x Z={HB})")
print(f"Total blocks mined: {len(visited_blocks)}")
print(f"Expected if no overlap: {n_stopes * blocks_per_stope}")
print(f"Total overlaps: {overlaps_found}")
if overlap_pairs:
    print(f"Example overlaps (RANK, block): {overlap_pairs[:5]}")
print("PASS — no overlaps" if overlaps_found == 0 else "FAIL — overlaps detected")
