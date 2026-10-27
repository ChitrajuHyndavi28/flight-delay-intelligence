"""Step 5a - Airport clustering (O6). Outputs -> results/clustering/.
Each origin airport is described by delay, cancellation, weather exposure/sensitivity and delay-cause features, then grouped with
K-Means (k chosen by silhouette, cross-checked with hierarchical clustering). Scope: non-October, known-weather airports, >= MIN_FLIGHTS flights."""
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from common import SEED, connect, outdir, set_plot_style, wx_class_sql

MIN_FLIGHTS, MIN_ADVERSE = 5000, 150
con = connect(); out = outdir("clustering"); plt = set_plot_style()
WX = wx_class_sql("w")
df = con.execute(f"""
WITH b AS (
  SELECT a.airport_iata_code AS airport, a.state, a.latitude, a.longitude, f.cancelled_flag, f.diverted_flag, f.arrival_delay_min,
         f.departure_delay_min, f.taxi_out_min, f.scheduled_departure_min // 60 AS hr, {WX} AS wx,
         f.late_aircraft_delay_min, f.weather_delay_min, f.airline_delay_min, f.air_system_delay_min, f.security_delay_min
  FROM fact_flight f JOIN dim_airport a ON a.airport_key = f.origin_airport_key
  JOIN dim_weather w ON w.weather_key = f.origin_weather_key JOIN dim_date d USING(date_key)
  WHERE d.month_number <> 10 AND f.origin_airport_key <> 0 AND f.origin_weather_key <> 0)
SELECT airport, any_value(state) AS state, any_value(latitude) AS lat, any_value(longitude) AS lon, count(*) AS flights,
  100.0 * avg(cancelled_flag) AS cancel_rate_pct,
  100.0 * avg(CASE WHEN cancelled_flag=0 AND diverted_flag=0 THEN (arrival_delay_min >= 15)::int END) AS delayed_pct,
  avg(CASE WHEN cancelled_flag=0 AND diverted_flag=0 THEN departure_delay_min END) AS mean_dep_delay_min,
  avg(CASE WHEN cancelled_flag=0 AND diverted_flag=0 THEN taxi_out_min END) AS mean_taxi_out_min,
  100.0 * avg((wx IN ('moderate','severe'))::int) AS adverse_wx_share_pct,
  count(*) FILTER (WHERE wx IN ('moderate','severe') AND cancelled_flag=0 AND diverted_flag=0) AS n_adverse,
  100.0 * avg(CASE WHEN wx IN ('moderate','severe') AND cancelled_flag=0 AND diverted_flag=0 THEN (arrival_delay_min>=15)::int END) AS delayed_adverse_pct,
  100.0 * avg(CASE WHEN wx='clear' AND cancelled_flag=0 AND diverted_flag=0 THEN (arrival_delay_min>=15)::int END) AS delayed_clear_pct,
  100.0 * avg((hr >= 18)::int) AS evening_share_pct,
  100.0 * sum(late_aircraft_delay_min) / nullif(sum(late_aircraft_delay_min+weather_delay_min+airline_delay_min+air_system_delay_min+security_delay_min),0) AS late_aircraft_share_pct,
  100.0 * sum(weather_delay_min) / nullif(sum(late_aircraft_delay_min+weather_delay_min+airline_delay_min+air_system_delay_min+security_delay_min),0) AS weather_cause_share_pct
FROM b GROUP BY airport HAVING count(*) >= {MIN_FLIGHTS}""").df()
df["weather_gap_pp"] = np.where(df.n_adverse >= MIN_ADVERSE, df.delayed_adverse_pct - df.delayed_clear_pct, np.nan)
print(f"{len(df)} airports with >= {MIN_FLIGHTS} flights; weather gap undefined for {df.weather_gap_pp.isna().sum()} (< {MIN_ADVERSE} adverse-weather flights) -> median-imputed")
df["weather_gap_pp"] = df.weather_gap_pp.fillna(df.weather_gap_pp.median())
df["log_flights"] = np.log10(df.flights)
FEATS = ["log_flights", "delayed_pct", "mean_dep_delay_min", "cancel_rate_pct", "mean_taxi_out_min", "adverse_wx_share_pct", "weather_gap_pp",
         "evening_share_pct", "late_aircraft_share_pct", "weather_cause_share_pct"]
X = StandardScaler().fit_transform(df[FEATS])

# ---- choose k --------------------------------------------------------------------------------------------------
sil = []
for k in range(2, 9):
    lab = KMeans(k, n_init=50, random_state=SEED).fit_predict(X); sil.append(dict(k=k, silhouette=silhouette_score(X, lab),
        inertia=KMeans(k, n_init=50, random_state=SEED).fit(X).inertia_))
sil = pd.DataFrame(sil); sil.to_csv(out / "k_selection.csv", index=False); print(sil.round(3).to_string(index=False))
# silhouette alone favours k=2 on small data; choose the best k in 3..6 so clusters are interpretable, and report both
K = int(sil[sil.k.between(3, 6)].sort_values("silhouette", ascending=False).iloc[0].k)
km = KMeans(K, n_init=100, random_state=SEED).fit(X); df["cluster"] = km.labels_
hc = AgglomerativeClustering(K, linkage="ward").fit_predict(X)
ari = adjusted_rand_score(km.labels_, hc); print(f"chosen k={K}, silhouette={silhouette_score(X, km.labels_):.3f}, ARI(K-Means vs Ward)={ari:.3f}")

# ---- stability: bootstrap ARI over features resampling of airports ---------------------------------------------------
rng = np.random.default_rng(SEED); aris = []
for _ in range(100):
    idx = rng.choice(len(X), len(X), replace=True); lab = KMeans(K, n_init=10, random_state=int(rng.integers(1e6))).fit(X[idx]).predict(X)
    aris.append(adjusted_rand_score(km.labels_, lab))
print(f"bootstrap stability ARI mean {np.mean(aris):.3f} (5th pct {np.percentile(aris, 5):.3f})")

# ---- profiles ---------------------------------------------------------------------------------------------------
prof = df.groupby("cluster")[FEATS[1:] + ["flights"]].mean().round(2); prof.insert(0, "airports", df.groupby("cluster").size())
prof["example_airports"] = df.sort_values("flights", ascending=False).groupby("cluster").airport.apply(lambda s: ", ".join(s.head(8)))
prof.to_csv(out / "cluster_profiles.csv"); print(prof.to_string())
df.sort_values(["cluster", "flights"], ascending=[True, False])[["airport", "state", "cluster", "flights"] + FEATS[1:]].round(2).to_csv(out / "airport_clusters.csv", index=False)
pd.DataFrame([dict(k=K, silhouette=silhouette_score(X, km.labels_), ari_kmeans_vs_ward=ari, bootstrap_ari_mean=np.mean(aris), bootstrap_ari_p5=np.percentile(aris, 5),
                   airports=len(df), min_flights=MIN_FLIGHTS)]).to_csv(out / "cluster_quality.csv", index=False)

# ---- plots ---------------------------------------------------------------------------------------------------------
pc = PCA(2, random_state=SEED).fit(X); Z = pc.transform(X); cols = plt.cm.tab10(np.arange(K))
fig, ax = plt.subplots(figsize=(9, 7))
for c in range(K):
    m = (df.cluster == c).to_numpy(); ax.scatter(Z[m, 0], Z[m, 1], s=30 + 12 * (df.log_flights[m] - df.log_flights.min()) ** 2, color=cols[c], alpha=.75, label=f"Cluster {c} (n={m.sum()})")
for i in df.sort_values("flights", ascending=False).index[:25]: ax.annotate(df.airport[i], (Z[df.index.get_loc(i), 0], Z[df.index.get_loc(i), 1]), fontsize=7)
ax.set_xlabel(f"PC1 ({100 * pc.explained_variance_ratio_[0]:.0f}% variance)"); ax.set_ylabel(f"PC2 ({100 * pc.explained_variance_ratio_[1]:.0f}% variance)")
ax.set_title("Airport clusters (K-Means, PCA projection)"); ax.legend(); fig.savefig(out / "airport_clusters_pca.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 6))
for c in range(K):
    m = df.cluster == c; ax.scatter(df.lon[m], df.lat[m], color=cols[c], s=18 + 6 * (df.log_flights[m] - df.log_flights.min()) ** 2, alpha=.8, label=f"Cluster {c}")
ax.set_xlim(-170, -65); ax.set_ylim(15, 72); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Airport clusters on the map"); ax.legend()
fig.savefig(out / "airport_clusters_map.png"); plt.close(fig)
z = pd.DataFrame(StandardScaler().fit_transform(df[FEATS[1:]]), columns=FEATS[1:]).groupby(df.cluster.to_numpy()).mean()
fig, ax = plt.subplots(figsize=(10, 4 + .4 * K)); im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(z.columns))); ax.set_xticklabels(z.columns, rotation=40, ha="right", fontsize=8); ax.set_yticks(range(K)); ax.set_yticklabels([f"Cluster {c}" for c in z.index])
ax.set_title("Cluster profile (z-scores of feature means)"); fig.colorbar(im); fig.savefig(out / "cluster_profile_heatmap.png"); plt.close(fig)
print("saved to", out)
