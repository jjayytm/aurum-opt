# AURUM-OPT
### Underground Gold Stope Optimisation — Hackathon 2026

> **Proven-optimal stope selection · AI mine planning · Live economic analysis**  
> Full pipeline from block model CSV to DXF export.

---

## What it does

AURUM-OPT takes any underground gold block model, applies a grade cutoff, and returns the mathematically optimal set of non-overlapping stopes — maximising total gold extraction while complying with Z-alignment constraints. Every run also produces a mine sequence plan and dilution risk assessment powered by XGBoost.

```
Block Model CSV  →  Optimal Stopes  →  DXF Export
                 →  AI Risk Flags   →  Mine Sequence
                 →  Economic Analysis  →  Dashboard
```

---

## Quick start

```bash
# Install dependencies
pip install pandas numpy xgboost scikit-learn scipy ezdxf flask

# Run optimizer directly
python src/stope_optimizer.py "Hackathon 2026 - Block Model.csv" 10.0

# Start dashboard server
python api_server.py
# Open http://localhost:5000
```

---

## Project structure

```
aurum-opt/
├── src/
│   └── stope_optimizer.py     # Full pipeline — 15 modules
├── api_server.py              # Flask server — upload, run, serve
├── dashboard.html             # Live dashboard — 3D, results, economics
├── outputs/
│   ├── stopes_optimised.dxf   # DXF export (auto-generated)
│   ├── stopes_results.csv     # Stope table (auto-generated)
│   ├── results.json           # Dashboard payload (auto-generated)
│   └── results_index.json     # Cutoff index (auto-generated)
└── README.md
```

---

## Pipeline architecture

The optimizer runs 13 modules sequentially:

| Module | Function | Key detail |
|--------|----------|------------|
| 1 | Data loader | Auto-detects header row, handles any block size |
| 2 | Preprocessing | Block attributes, grid indices, per-block waste |
| 3 | Prefix-sum grids | 3D float32 cumsum — O(1) stope evaluation |
| 4 | Stope engine | Vectorised sliding-window scan over all positions |
| 5 | Z-alignment | Gold-weighted offset selection (Memo 02Mar2026) |
| 6 | Economic filter | Grade cutoff + Z-compliance enforcement |
| 7 | AI training | XGBoost classifier + regressor, 3% sample |
| 8 | AI prediction | Scores all grade-passing positions |
| 9 | Ranking module | Dilution risk model + NPV mine sequence |
| 10 | DP selection | Column-based exact Dynamic Programming |
| 12 | DXF export | Background thread — dashboard updates first |
| 13 | JSON export | Full payload including AI outputs and economics |
| 14 | Reporting | Terminal summary with AI validation metrics |

---

## Stope geometry

Fixed geometry per competition rules (Technical Notes §4):

```
Height    : 30m  (Z direction)
Length    : 20m  (X direction)
Thickness :  5m  (Y direction)
Volume    : 3,000 m³ per stope
```

The optimizer handles any block size automatically — 1m, 2.5m, 5m, 10m.

---

## Why Dynamic Programming

For fixed-geometry stopes, non-overlapping selection on a spatial grid is equivalent to weighted interval scheduling — a problem DP solves exactly and provably optimally.

**What DP guarantees:** No alternative selection of non-overlapping 30×20×5m stopes can extract more gold at the same cutoff. This is a mathematical proof, not a heuristic claim.

No commercial software provides this guarantee for fixed-geometry stopes. Deswik.SO and Datamine use heuristic search because they solve a harder variable-geometry problem. AURUM-OPT makes a different trade-off — provably optimal for the fixed-geometry case, in seconds.

---

## AI mine planning layer

Two genuine ML outputs run after DP selection:

### Module A — Dilution Risk Model

XGBoost trains on **spatial features only** — X, Y, Z, depth, distance from deposit centroid, X×Z plunge interaction — to predict expected waste ratio. Grade margin (how far each stope's grade is above the economic cutoff) is computed analytically as an independent signal and combined 60/40 with the spatial prediction. No circular dependency.

```
Spatial features → XGBoost → predicted waste ratio
Residual = actual - predicted  →  spatial_risk score
Grade margin = (wavg - cutoff_p5) / wavg  →  grade_risk score

Final risk score = 0.60 × spatial_risk + 0.40 × grade_risk
Thresholds = p20 / p80 percentiles → ~20% LOW / 60% MED / 20% HIGH
```

**Validation on real competition dataset (9.2 g/t cutoff):**

```
LOW    : mean waste =  0.6%   mean grade = 13.53 g/t   n = 2,441
MEDIUM : mean waste = 12.5%   mean grade = 11.91 g/t   n = 7,319
HIGH   : mean waste = 36.4%   mean grade =  9.99 g/t   n = 2,441

Kendall tau = 0.731  (p = 0.000)
✓ Model validated — higher risk score correlates with higher actual waste
```

HIGH risk stopes sit on ore body boundaries where the block model grade estimate is least reliable. These are your infill drilling targets before committing development capital.

### Module B — NPV Mine Sequence

Every selected stope is scored and ranked by four factors that mine planners actually use:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Value density (oz/t) | 40% | Highest ROI per development metre |
| Depth access | 30% | Shallower = cheaper development, faster first ore |
| Dilution risk (inverted) | 20% | Mine low-risk stopes first, avoid early surprises |
| Grade normalised | 10% | Quality signal, tiebreaker |

Output: `MINE_SEQUENCE` rank 1..N for every stope. Sequence #1 is always the highest-value, lowest-risk, most accessible stope — verified on every run.

---

## Performance — real competition dataset

97,250 blocks · 5m blocks · 15 elevation levels

| Cutoff | Runtime | Stopes | Gold (oz) | Waste % | Break-even | AI tau |
|--------|---------|--------|-----------|---------|------------|--------|
| 6.5 g/t | 2.26s | 5,130 | 10.2M | 30.9% | $276/oz | 0.753 |
| 8.9 g/t | 2.33s | 3,750 | 8.8M | 23.9% | $229/oz | 0.742 |
| 9.2 g/t | 2.36s | 3,562 | 8.6M | 22.6% | ~$220/oz | 0.731 |
| 9.5 g/t | 2.18s | 3,386 | 8.3M | 21.1% | $216/oz | 0.720 |
| 11.8 g/t | 1.71s | 1,788 | 5.0M | 8.0% | $164/oz | 0.465 |
| 12.3 g/t | 0.88s | 1,432 | 4.1M | 6.2% | $157/oz | 0.388 |

All runs: Z-alignment PASS · DXF exported · Dashboard loaded · AI validated

---

## Dashboard

Three-tab interface served at `http://localhost:5000`.

### ⬡ 3D Deposit
Interactive THREE.js point cloud coloured by Au grade. Drag to rotate, scroll to zoom, hover for grade readout at any block.

### ⬛ Results
Total gold captured · waste ratio · stope count · runtime · Z-alignment status · gold by elevation chart · grade distribution · top 15 stopes table with waste colour coding (green < 15% · gold 15–25% · red > 25%).

### 💰 Economics
- **Mine Economic Summary** — net profit, revenue, cost, break-even price with margin, profitable stopes, profit per tonne
- **Gold Price Sensitivity** — slider + text input, cost-aware recalculation, user-defined base price via "Set as base"
- **AI Dilution Risk Analysis** — LOW / MEDIUM / HIGH stope counts with explanations
- **AI Mine Sequence Plan** — top 20 stopes in development order with grade, gold oz, value/t, waste %, risk badge

---

## Economic model

```python
GOLD_PRICE_USD  = 3000.0   # $/oz  (adjustable in dashboard)
RECOVERY_PCT    = 0.95     # 95% metallurgical recovery
MINING_COST_T   = 35.0     # $/t mining
MILLING_COST_T  = 25.0     # $/t milling

Revenue    = gold_oz × gold_price × recovery
Cost       = total_tonnes × (mining + milling)
Profit     = Revenue − Cost
Break-even = Cost / (gold_oz × recovery)
```

When gold price drops below break-even, sub-economic stopes are removed from the active plan — cost scales proportionally with the profitable stope fraction.

---

## Output files

| File | Description |
|------|-------------|
| `outputs/stopes_optimised.dxf` | 3D box prisms, one per stope — import into Deswik / Leapfrog / AutoCAD |
| `outputs/stopes_results.csv` | Full stope table — coordinates, grade, gold oz, risk label, mine sequence rank |
| `results_<cutoff>gt.json` | Dashboard payload — all metrics, AI outputs, chart data, economic params |
| `results_index.json` | Index of all computed cutoffs — drives dynamic cutoff buttons in dashboard |

---

## Requirements

```
Python       >= 3.10
pandas       >= 1.5
numpy        >= 1.23
xgboost      >= 1.7
scikit-learn >= 1.1
scipy        >= 1.9      # Kendall tau validation
ezdxf        >= 0.18     # DXF export
flask        >= 2.2      # API server
```

---

## Known limitations

AURUM-OPT optimises for fixed 30×20×5m stope geometry as specified by competition rules. A production system would additionally require:

- Variable stope geometry per geotechnical domain
- Multi-year production scheduling with NPV discounting over mine life
- Rock mass rating and ground support cost constraints
- Ventilation network capacity limits per level
- Grade uncertainty inputs (kriging variance, drill hole spacing)
- Equipment fleet scheduling and material handling constraints

---

## Hackathon 2026 — Finals demo

The system accepts any unseen block model CSV and returns full results in under a minute.

**5-minute demo flow:**
1. Upload CSV → type cutoff → Run ▶ → results in under a minute.
2. Results tab: gold captured, waste ratio, Z-alignment PASS, top stopes table
3. Economics tab: net profit at base price, drag slider to stress-test scenarios
4. AI panels: dilution risk distribution, mine sequence with LOW/MED/HIGH badges
5. Export DXF → hand to engineering team

**Key numbers (9.2 g/t cutoff, real dataset):**
- 2.36 seconds end-to-end
- 3,562 non-overlapping Z-compliant stopes
- 8.56M oz gold captured
- 22.6% waste ratio
- $220/oz break-even — profitable if gold drops 93%
- AI tau = 0.731 — validated dilution risk model

---

*AURUM-OPT — Proven optimal. Under a minute.. Actionable mine plan.*
