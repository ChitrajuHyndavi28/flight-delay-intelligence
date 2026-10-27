"""Takes a screenshot of every dashboard page/tab with headless Chrome and assembles them into DASHBOARD_SCREENSHOTS.pdf.
Needs the dashboard server running (bash run_dashboard.sh) and Google Chrome installed."""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "screenshots"; OUT.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; BASE = "http://localhost:8765/"; W = 1440
PAGES = [("overview", "Overview", ""), ("t60/results", "T-60 · Performance", ""), ("t60/explain", "T-60 · Weather and drivers", ""), ("t60/predict", "T-60 · Live predictor (a real Nov–Dec flight)", "?autorun"), ("t60/paper", "T-60 · Against the base paper", ""),
         ("dayahead/results", "Day-ahead · Performance", ""), ("dayahead/explain", "Day-ahead · Weather and features", ""), ("dayahead/paper", "Day-ahead · Against the base paper", ""),
         ("postdep/results", "Post-departure · Performance", ""), ("postdep/paper", "Post-departure · Against the base paper", ""),
         ("evaluation/scenarios", "Metric evaluation · Scenarios", ""), ("evaluation/models", "Metric evaluation · Models", ""), ("evaluation/curves", "Metric evaluation · Curves", ""), ("evaluation/paper", "Metric evaluation · Against the base paper", ""), ("evaluation/tuning", "Metric evaluation · Tuning", ""),
         ("explore/time", "Explore · Time and airlines", ""), ("explore/weather", "Explore · Weather and causes", ""), ("explore/prop", "Explore · Delay propagation", ""), ("explore/airports", "Explore · Airports and cancellations", ""), ("about", "About and limits", "")]

def shot(route, q, h, path):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--force-prefers-reduced-motion", f"--window-size={W},{h}", "--virtual-time-budget=12000", f"--screenshot={path}", f"{BASE}{q}#/{route}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def content_bottom(path):
    a = np.asarray(Image.open(path).convert("RGB"))[:, 290:W - 30].astype(int); bg = a[5, 5]; rows = np.where((np.abs(a - bg).sum(axis=2) > 12).any(axis=1))[0]; return int(rows.max()) if len(rows) else 0

imgs = []
for route, title, q in PAGES:
    tmp = OUT / "_tmp.png"; h = 3800; shot(route, q, h, tmp); b = content_bottom(tmp)
    if b > h - 80: h = b + 1500; shot(route, q, h, tmp); b = content_bottom(tmp)
    final = OUT / (route.replace("/", "_") + ".png"); shot(route, q, max(900, b + 70), final); im = Image.open(final).convert("RGB"); imgs.append((title, im)); print(f"{title:55s} {im.size}", flush=True)
(OUT / "_tmp.png").unlink(missing_ok=True)

# cover page
fig = plt.figure(figsize=(14.4, 9), dpi=100); fig.patch.set_facecolor("#f5f6f8"); fig.text(0.07, 0.78, "Flight Delay Intelligence", fontsize=40, family="serif", color="#161b26"); fig.text(0.07, 0.71, "Dashboard screenshots", fontsize=22, color="#5d6678")
fig.text(0.07, 0.65, "2015 US flights · T-60, day-ahead and post-departure prediction · weather impact · base-paper comparison", fontsize=13, color="#5d6678")
for i, (t, _) in enumerate(imgs): fig.text(0.07 + 0.43 * (i // 10), 0.56 - 0.045 * (i % 10), f"{i + 2:>2}   {t}", fontsize=12, color="#161b26")
fig.text(0.07, 0.05, "Light theme, 1440 px wide. Captured from the local portal at http://localhost:8765", fontsize=10, color="#5d6678"); fig.savefig(OUT / "_cover.png"); plt.close(fig)
pages = [Image.open(OUT / "_cover.png").convert("RGB")] + [im for _, im in imgs]
pdf = ROOT / "DASHBOARD_SCREENSHOTS.pdf"; pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150.0); (OUT / "_cover.png").unlink(missing_ok=True); print("saved", pdf, f"({pdf.stat().st_size / 1e6:.1f} MB, {len(pages)} pages)")
