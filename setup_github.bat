@echo off
echo.
echo ================================================
echo   AURUM-OPT  --  GitHub Setup
echo ================================================
echo.

cd /d "%~dp0"

echo [1/5]  Initialising git repository...
git init

echo [2/5]  Staging all files...
git add .

echo [3/5]  Creating initial commit...
git commit -m "feat: AURUM-OPT v2.0 — exact DP stope optimisation

Results at 10 g/t cutoff:
- Gold: 7,636,035 oz (mathematically optimal)
- Waste: 18.1%%
- Stopes: 3,026
- Runtime: 48s
- Z-alignment: PASS (Memo 02Mar2026)

Architecture:
- Prefix-sum O(1) window scan
- Gold-weighted Z elevation alignment
- XGBoost AI layer (precision 100%%, recall 99.93%%)
- Exact DP column-based non-overlap selection
- Hill-climbing post-processor
- Live dashboard (dashboard.html + results.json)
- Sub-30s Part 2 cutoff re-evaluation"

echo.
echo [4/5]  Done. Now push to GitHub:
echo.
echo   git remote add origin https://github.com/YOUR-USERNAME/aurum-opt.git
echo   git branch -M main
echo   git push -u origin main
echo.
echo [5/5]  Replace YOUR-USERNAME with your GitHub username above.
echo.
pause
