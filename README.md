# AURUM-OPT — Render Deployment

## Folder structure

```
aurum-render/
├── server.py                          # Flask API server
├── requirements.txt                   # Python dependencies
├── render.yaml                        # Render.com config
├── src/
│   └── stope_optimizer.py             # Your optimizer (unchanged)
├── static/
│   ├── dashboard.html                 # Live dashboard
│   └── points3d.json                  # 3D block model points
└── Hackathon 2026 - Block Model.csv   # Upload manually (see below)
```

## Deploy to Render — step by step

### 1. Push to GitHub

```bash
cd aurum-render
git init
git add .
git commit -m "AURUM-OPT deploy"
git remote add origin https://github.com/YOUR-USERNAME/aurum-opt-deploy.git
git push -u origin main
```

### 2. Upload CSV to GitHub

The block model CSV is too large to track in git normally.
Add it with git-lfs OR just commit it directly (it's 5MB, fine for GitHub):

```bash
git add "Hackathon 2026 - Block Model.csv"
git commit -m "add block model"
git push
```

### 3. Create Render web service

1. Go to https://render.com → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — click Deploy
4. Wait ~3 minutes for first build

### 4. Your live URL

Render gives you: `https://aurum-opt.onrender.com`

Open that URL — dashboard loads, judges can type any cutoff and hit Run.

## How the Run button works

1. Judge types `9.4` in the input box → clicks Run
2. Dashboard POSTs to `/api/run` → server starts optimizer in background thread
3. Dashboard polls `/api/status` every 2 seconds → shows live progress bar
4. When done → new `9.4 g/t` button appears automatically → KPIs update
5. Export DXF button downloads the result

## Important: free tier sleep

Render free tier sleeps after 15 minutes of inactivity.
First request after sleep takes ~30 seconds to wake up.

**Fix:** Open the dashboard URL 2 minutes before your presentation.
The optimizer itself still takes 14-48 seconds to run after the server is awake.
