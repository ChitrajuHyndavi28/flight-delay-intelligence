"""Diagrams for the project report: system architecture, star schema, prediction timeline. -> results/report_figs/"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "report_figs"; OUT.mkdir(parents=True, exist_ok=True)
INK, LINE = "#1c2333", "#5d6678"

def box(ax, x, y, w, h, title, sub="", fc="#eef3fb", ec="#2457d6", fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, family="DejaVu Sans")
    if sub: ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 2.5, color=LINE, family="DejaVu Sans")
def arrow(ax, x1, y1, x2, y2, c=LINE): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=c, shrinkA=0, shrinkB=0))

# 1. architecture ----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
box(ax, 0.2, 6.1, 2.9, 1.0, "DOT on-time data 2015", "5.8M flights · tail numbers", "#fff4e5", "#cf7a0a"); box(ax, 3.55, 6.1, 2.9, 1.0, "NOAA GHCN-Daily", "daily weather · stations", "#fff4e5", "#cf7a0a"); box(ax, 6.9, 6.1, 2.9, 1.0, "Lookup files", "airlines.csv · airports.csv", "#fff4e5", "#cf7a0a")
box(ax, 0.2, 4.6, 9.6, 1.0, "ETL pipeline (Python)", "clean · map airports to stations · derive keys · load", "#eaf6f0", "#16705f")
box(ax, 0.2, 3.1, 9.6, 1.0, "Star-schema data warehouse (DuckDB)", "fact_flight + 5 dimensions · derived aircraft-chain (propagation) tables", "#eaf6f0", "#16705f")
row = [("OLAP and audit", "roll-up · slice\ndice · pivot"), ("Root cause,\nweather", "controlled gaps\nodds ratios"), ("Delay\npropagation", "tail-number chains\nbuffer absorption"), ("Pattern\ndiscovery", "K-Means clusters\nassociation rules"), ("Feature tables", "T-60 · day-ahead\npost-departure")]
for i, (t, sb) in enumerate(row): box(ax, 0.2 + i * 1.96, 1.6, 1.82, 1.2, t, sb, "#f3effb" if i < 4 else "#eef3fb", "#7b3fd4" if i < 4 else "#2457d6", 9.4)
box(ax, 5.9, 0.15, 3.9, 1.0, "Models and evaluation", "11 models · MLP · ensemble · PSO · SHAP", "#eef3fb", "#2457d6"); box(ax, 0.2, 0.15, 4.6, 1.0, "Dashboard, API and report", "local portal · live predictor · Word report", "#fdeef2", "#9a3b6b")
for x in (1.65, 5.0, 8.35): arrow(ax, x, 6.1, x, 5.6)
arrow(ax, 5.0, 4.6, 5.0, 4.1)
for i in range(5): arrow(ax, 1.11 + i * 1.96, 3.1, 1.11 + i * 1.96, 2.8)
arrow(ax, 9.0 - 0.05, 1.6, 9.0 - 0.05, 1.15); arrow(ax, 5.9, 0.65, 4.8, 0.65); arrow(ax, 1.11, 1.6, 1.11, 1.15)
fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 2. star schema -------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")
def table(x, y, w, h, name, cols, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=ec, lw=1.8)); ax.add_patch(FancyBboxPatch((x, y + h - 0.45), w, 0.45, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h - 0.22, name, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    for i, c in enumerate(cols): ax.text(x + 0.12, y + h - 0.75 - i * 0.3, c, fontsize=8.6, va="center", color=INK, family="DejaVu Sans Mono")
table(3.5, 1.1, 3.0, 5.5, "fact_flight", ["flight_key (PK)", "date_key (FK)", "airline_key (FK)", "origin_airport_key (FK)", "destination_airport_key (FK)", "route_key (FK)", "origin_weather_key (FK)", "destination_weather_key (FK)", "tail_number · flight_number", "scheduled / actual times", "departure_delay_min", "arrival_delay_min", "cancelled_flag · diverted_flag", "5 delay-cause minutes"], "#e3ecfb", "#2457d6")
table(0.2, 4.6, 2.9, 2.0, "dim_date", ["date_key (PK)", "full_date · month · quarter", "day_of_week · season", "weekend_flag · holiday_flag"], "#fff1dc", "#cf7a0a")
table(0.2, 0.9, 2.9, 1.6, "dim_airline", ["airline_key (PK)", "airline_iata_code", "airline_name"], "#fff1dc", "#cf7a0a")
table(3.6, 6.95, 2.8, 1.3, "dim_route", ["route_key (PK)", "origin · destination · band"], "#fff1dc", "#cf7a0a")
table(6.9, 4.6, 2.9, 1.8, "dim_airport", ["airport_key (PK)", "iata_code · name · city", "state · latitude · longitude"], "#fff1dc", "#cf7a0a")
table(6.9, 0.9, 2.9, 2.4, "dim_weather", ["weather_key (PK)", "airport · weather_date", "precip_mm · snowfall_mm", "snow_depth · tmax · tmin", "avg_wind_ms · station"], "#fff1dc", "#cf7a0a")
for (x1, y1, x2, y2) in [(3.1, 5.6, 3.5, 5.0), (3.1, 1.9, 3.5, 2.3), (5.0, 6.95, 5.0, 6.62), (6.9, 5.5, 6.5, 5.0), (6.9, 2.1, 6.5, 2.4)]: arrow(ax, x1, y1, x2, y2, "#2457d6")
ax.text(5, 0.35, "Star schema: one fact table (one row per scheduled flight) and five dimensions", ha="center", fontsize=10, color=LINE)
fig.savefig(OUT / "star_schema.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 3. prediction moments ---------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.7)); ax.set_xlim(-0.3, 10.3); ax.set_ylim(0, 3.7); ax.axis("off"); ax.plot([0.3, 9.7], [1.5, 1.5], color=INK, lw=2.4)
pts = [(1.0, "Day before", "Day-ahead\nprediction", "#cf7a0a"), (4.0, "T-60", "60 min before\nscheduled departure", "#2457d6"), (6.5, "Take-off", "own departure delay\nbecomes known", "#7b3fd4"), (9.0, "Arrival", "delayed if 15+ min\nlate (the label)", "#17803d")]
for x, a, b, c in pts: ax.scatter([x], [1.5], s=190, color=c, zorder=3); ax.text(x, 2.15, a, ha="center", fontsize=12, fontweight="bold", color=c); ax.text(x, 0.75, b, ha="center", va="top", fontsize=9.5, color=INK)
ax.annotate("", xy=(4.0, 2.9), xytext=(1.0, 2.9), arrowprops=dict(arrowstyle="<->", color=LINE)); ax.text(2.5, 3.05, "schedule, airports, weather", ha="center", fontsize=9, color=LINE)
ax.annotate("", xy=(6.5, 3.15), xytext=(4.0, 3.15), arrowprops=dict(arrowstyle="<->", color=LINE)); ax.text(5.25, 3.3, "+ inbound aircraft status and delay", ha="center", fontsize=9, color=LINE)
fig.savefig(OUT / "prediction_moments.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
