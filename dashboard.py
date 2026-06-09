import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------
# SAYFA AYARLARI
# ---------------------------------------------------

st.set_page_config(
    page_title="Shodan Siber Risk Analizi",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# ÜLKE İSİMLERİ & KODLAR
# ---------------------------------------------------

country_names = {
    "TR": "Türkiye",
    "US": "Amerika",
    "DE": "Almanya",
    "FR": "Fransa",
    "RU": "Rusya",
    "CN": "Çin",
    "GB": "İngiltere",
    "NL": "Hollanda",
    "JP": "Japonya",
    "IN": "Hindistan",
    "BR": "Brezilya"
}

country_map_iso3 = {
    "TR": "TUR", "US": "USA", "DE": "DEU", "FR": "FRA",
    "RU": "RUS", "CN": "CHN", "GB": "GBR", "NL": "NLD",
    "JP": "JPN", "IN": "IND", "BR": "BRA"
}

internet_users_m = {
    "TR": 67.8, "US": 311.3, "DE": 75.5, "FR": 60.4,
    "RU": 127.4, "CN": 1050.0, "GB": 65.0, "NL": 17.1,
    "JP": 102.5, "IN": 820.0, "BR": 181.8
}

# ---------------------------------------------------
# CYBER CSS
# ---------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #050816;
    color: #e2e8f0;
    font-family: 'Rajdhani', 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 98%;
}

h1, h2, h3 {
    color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 0.05em;
}

.metric-card {
    background: linear-gradient(145deg, #0d1324, #111827);
    padding: 22px 24px;
    border-radius: 12px;
    border: 1px solid rgba(0,229,255,0.18);
    box-shadow: 0 0 24px rgba(0,229,255,0.08),
                inset 0 0 40px rgba(0,0,0,0.3);
    text-align: center;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
}

.metric-card.danger::before { background: linear-gradient(90deg, transparent, #ff2d55, transparent); }
.metric-card.warn::before   { background: linear-gradient(90deg, transparent, #ffb800, transparent); }
.metric-card.ok::before     { background: linear-gradient(90deg, transparent, #39ff7a, transparent); }

.metric-card h1 {
    color: #00e5ff;
    font-size: 2rem;
    font-family: 'Share Tech Mono', monospace;
    margin: 6px 0 0 0;
}

.metric-card.danger h1 { color: #ff2d55 !important; }
.metric-card.warn h1   { color: #ffb800 !important; }
.metric-card.ok h1     { color: #39ff7a !important; }

.metric-card h3 {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
}

.badge-high    { background: rgba(255,45,85,0.18);   color: #ff2d55; border: 1px solid rgba(255,45,85,0.3); }
.badge-med     { background: rgba(255,184,0,0.18);   color: #ffb800; border: 1px solid rgba(255,184,0,0.3); }
.badge-low     { background: rgba(57,255,122,0.12);  color: #39ff7a; border: 1px solid rgba(57,255,122,0.25); }
.badge-sus     { background: rgba(255,45,85,0.12);   color: #ff6b8a; border: 1px solid rgba(255,45,85,0.25); }
.badge-normal  { background: rgba(57,255,122,0.08);  color: #39ff7a; border: 1px solid rgba(57,255,122,0.2); }
.badge-cve     { background: rgba(255,100,0,0.15);   color: #ff6400; border: 1px solid rgba(255,100,0,0.3); }

.ai-block {
    background: #0d1324;
    border-left: 3px solid #00e5ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 14px;
    font-family: 'Rajdhani', sans-serif;
}

.ai-block.high { border-left-color: #ff2d55; }
.ai-block.med  { border-left-color: #ffb800; }
.ai-block.low  { border-left-color: #39ff7a; }
.ai-block.cve  { border-left-color: #ff6400; }

.ai-country           { font-size: 1rem; font-weight: 700; color: #00e5ff; margin-bottom: 4px; }
.ai-country.high      { color: #ff2d55; }
.ai-country.med       { color: #ffb800; }
.ai-country.cve-title { color: #ff6400; }

.ai-text {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    line-height: 1.6;
}

.panel {
    background: #0d1324;
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.panel-title {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    margin-bottom: 14px;
}

.process-step {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(0,229,255,0.08);
}

.step-num {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: #111827;
    border: 1px solid #00e5ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: #00e5ff;
    font-family: 'Share Tech Mono', monospace;
    flex-shrink: 0;
}

.step-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.step-item  { font-size: 0.85rem; color: rgba(255,255,255,0.5); padding: 2px 0; }
.step-item::before { content: '› '; color: #00e5ff; }

.topbar {
    background: #0d1324;
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.chip {
    display: inline-block;
    background: #111827;
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #00e5ff;
    margin: 3px;
    letter-spacing: 0.06em;
    font-family: 'Share Tech Mono', monospace;
}

.code-block {
    background: #070d1a;
    border: 1px solid rgba(255,100,0,0.2);
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.8;
    white-space: pre;
    overflow-x: auto;
    margin-bottom: 14px;
}

.code-keyword  { color: #ff6400; }
.code-string   { color: #39ff7a; }
.code-comment  { color: rgba(255,255,255,0.3); font-style: italic; }
.code-func     { color: #00e5ff; }
.code-var      { color: #ffb800; }

.alert-critical {
    background: rgba(255,45,85,0.1);
    border: 1px solid rgba(255,45,85,0.3);
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 0.85rem;
    color: #ff2d55;
    margin-bottom: 16px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.04em;
}

section[data-testid="stSidebar"] {
    background-color: #0a1020;
    border-right: 1px solid rgba(0,229,255,0.1);
}

button[data-baseweb="tab"] {
    background-color: #0d1324 !important;
    color: rgba(255,255,255,0.5) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0,229,255,0.1) !important;
    margin-right: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #111827 !important;
    color: #00e5ff !important;
    border-color: rgba(0,229,255,0.35) !important;
}

div[data-baseweb="tab-highlight"] { background-color: #00e5ff !important; }

div[data-testid="metric-container"] {
    background: #0d1324;
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 10px;
    padding: 12px;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
}

.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #39ff7a;
    animation: blink 1.4s ease-in-out infinite;
    margin-right: 6px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TOPBAR
# ---------------------------------------------------

st.markdown("""
<div class="topbar">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:28px;height:28px;background:#00e5ff;
             clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);"></div>
        <span style="font-size:1.1rem;font-weight:700;color:#00e5ff;
               letter-spacing:0.12em;text-transform:uppercase;
               font-family:'Rajdhani',sans-serif;">
            ShodanIntel · Siber Risk Analizi
        </span>
    </div>
    <div style="font-size:0.75rem;font-family:'Share Tech Mono',monospace;color:#39ff7a;">
        <span class="pulse-dot"></span> SİSTEM AKTİF · CANLI ANALİZ
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# VERİYİ OKU
# ---------------------------------------------------

df = pd.read_csv("raw_data.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown("""
<div style="text-align:center; padding:10px 0 20px;">
    <div style="font-size:1.2rem;font-weight:700;color:#00e5ff;
         letter-spacing:0.15em;text-transform:uppercase;
         font-family:'Rajdhani',sans-serif;">⚙️ FİLTRELER</div>
</div>
""", unsafe_allow_html=True)

selected_country = st.sidebar.multiselect(
    "Ülke Seç",
    options=df["country"].unique(),
    default=df["country"].unique(),
    format_func=lambda x: country_names.get(x, x)
)

filtered_df = df[df["country"].isin(selected_country)]

st.sidebar.markdown("---")
st.sidebar.info("Dashboard Shodan API verileri kullanılarak oluşturulmuştur.")

# ---------------------------------------------------
# PIVOT TABLE
# ---------------------------------------------------

pivot = filtered_df.pivot(
    index="country",
    columns="service",
    values="total"
).fillna(0)

# ---------------------------------------------------
# RISK SCORE HESAPLAMA
# ---------------------------------------------------

risk_weights = {
    "port:22":       3,
    "port:3389":     5,
    "port:21":       4,
    "port:445":      5,
    "mongodb":       4,
    "redis":         4,
    "elasticsearch": 3
}

pivot["risk_score"] = 0

for service, weight in risk_weights.items():
    if service in pivot.columns:
        pivot["risk_score"] += pivot[service] * weight

# ---------------------------------------------------
# NORMALIZASYON
# ---------------------------------------------------

features = pivot.drop(columns=["risk_score"])
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ---------------------------------------------------
# K-MEANS CLUSTERING
# ---------------------------------------------------

kmeans           = KMeans(n_clusters=3, random_state=42, n_init=10)
pivot["cluster"] = kmeans.fit_predict(X_scaled)

cluster_risk    = pivot.groupby("cluster")["risk_score"].mean()
sorted_clusters = cluster_risk.sort_values().index.tolist()

cluster_map = {
    sorted_clusters[0]: "Düşük Risk",
    sorted_clusters[1]: "Orta Risk",
    sorted_clusters[2]: "Yüksek Risk"
}

pivot["risk_level"] = pivot["cluster"].map(cluster_map)

# ---------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------

iso = IsolationForest(contamination=0.2, random_state=42)
pivot["anomaly"] = iso.fit_predict(X_scaled)

pivot["anomaly_label"] = pivot["anomaly"].map({
    1:  "Normal",
    -1: "Şüpheli"
})

# ---------------------------------------------------
# DİNAMİK AI YORUM SİSTEMİ
# ---------------------------------------------------

high_threshold   = pivot["risk_score"].quantile(0.80)
median_threshold = pivot["risk_score"].median()


def generate_comment(row):
    score = row["risk_score"]
    rdp_flag, smb_flag, ssh_flag = "", "", ""

    if "port:3389" in pivot.columns and row.get("port:3389", 0) > 0:
        rdp_flag = " RDP exposure brute-force ve kimlik bilgisi sızdırma riskini artırmaktadır."
    if "port:445" in pivot.columns and row.get("port:445", 0) > 0:
        smb_flag = " SMB exposure EternalBlue türevi saldırılara karşı açık konumdadır."
    if "port:22" in pivot.columns and row.get("port:22", 0) > 0:
        ssh_flag = " SSH exposure birincil tehdit vektörünü oluşturmaktadır."

    if score > high_threshold:
        return (
            "Kritik saldırı yüzeyi tespit edilmiştir. "
            "Uzaktan erişim servisleri yüksek risk oluşturmaktadır."
            + rdp_flag + smb_flag
        )
    elif score > median_threshold:
        return (
            "Orta seviyede risk gözlemlenmiştir."
            + ssh_flag
            + " Servis güvenlik yapılandırmaları ve erişim kontrolü politikaları gözden geçirilmelidir."
        )
    else:
        return (
            "Görece düşük risk seviyesi tespit edilmiştir. "
            "Mevcut güvenlik önlemleri yeterli görünmektedir."
        )


pivot["ai_comment"] = pivot.apply(generate_comment, axis=1)

# ---------------------------------------------------
# ÜLKE BAŞINA EXPOSURE ORANI
# ---------------------------------------------------

pivot["internet_users_m"] = pivot.index.map(internet_users_m)
pivot["exposure_per_user"] = (
    pivot["risk_score"] / (pivot["internet_users_m"] * 1_000_000)
).round(4)

# ---------------------------------------------------
# ZAMAN DAMGALI CSV KAYDI
# ---------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_dir   = "data_history"
os.makedirs(csv_dir, exist_ok=True)

raw_csv_path = os.path.join(csv_dir, f"raw_data_{timestamp}.csv")
df.to_csv(raw_csv_path, index=False)

risk_save = pivot[["risk_score", "risk_level", "anomaly_label",
                    "exposure_per_user"]].copy()
risk_save["timestamp"] = timestamp
risk_csv_path = os.path.join(csv_dir, f"risk_scores_{timestamp}.csv")
risk_save.to_csv(risk_csv_path)

ai_report = pivot[["risk_score", "risk_level",
                    "anomaly_label", "ai_comment"]].copy()
ai_report["timestamp"] = timestamp
ai_csv_path = os.path.join(csv_dir, f"ai_risk_report_{timestamp}.csv")
ai_report.to_csv(ai_csv_path)

# ---------------------------------------------------
# ZAMAN SERİSİ — geçmiş CSV'lerden risk trendi
# ---------------------------------------------------

history_files = sorted([
    f for f in os.listdir(csv_dir)
    if f.startswith("risk_scores_") and f.endswith(".csv")
])

trend_records = []
for fname in history_files:
    try:
        tmp = pd.read_csv(os.path.join(csv_dir, fname), index_col=0)
        if "risk_score" in tmp.columns and "timestamp" in tmp.columns:
            ts = tmp["timestamp"].iloc[0]
            for country, row_data in tmp.iterrows():
                trend_records.append({
                    "timestamp":  ts,
                    "country":    country,
                    "risk_score": row_data["risk_score"]
                })
    except Exception:
        pass

trend_df = pd.DataFrame(trend_records)

# ---------------------------------------------------
# CVE VERİSİ
# ---------------------------------------------------

cve_queries = {
    "Log4Shell (CVE-2021-44228)":  "vuln:CVE-2021-44228",
    "ProxyLogon (CVE-2021-26855)": "vuln:CVE-2021-26855",
    "EternalBlue (CVE-2017-0144)": "vuln:CVE-2017-0144",
    "Heartbleed (CVE-2014-0160)":  "vuln:CVE-2014-0160",
}

cve_csv = "cve_data.csv"
if os.path.exists(cve_csv):
    cve_df = pd.read_csv(cve_csv)
else:
    cve_df = pd.DataFrame({
        "cve":   list(cve_queries.keys()),
        "total": [245000, 189000, 134000, 98000]
    })

# ---------------------------------------------------
# GLOBAL ALERT
# ---------------------------------------------------

max_risk = pivot["risk_score"].max()
if max_risk > 10_000_000:
    st.markdown("""
    <div class="alert-critical">
        🚨 KRİTİK UYARI: Global exposure seviyesi kritik eşiği aştı —
        yüksek riskli ülkeler acil güvenlik denetimi gerektirmektedir.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# MAP DATAFRAME
# ---------------------------------------------------

map_df = pivot.reset_index().copy()
map_df["country_name"] = map_df["country"].map(country_names)
map_df["iso3"]         = map_df["country"].map(country_map_iso3)
map_df["risk_log"]     = np.log10(map_df["risk_score"] + 1)

# ---------------------------------------------------
# DÜNYA HARİTASI
# ---------------------------------------------------

fig_map = px.choropleth(
    map_df,
    locations="iso3",
    color="risk_log",
    hover_name="country_name",
    hover_data={"risk_score": ":,.0f", "risk_level": True, "risk_log": False},
    color_continuous_scale=[
        [0.0,  "#0a1628"],
        [0.25, "#0f3460"],
        [0.5,  "#533483"],
        [0.75, "#e94560"],
        [1.0,  "#ff2d55"]
    ],
    title="🌍 Global Siber Risk Haritası"
)

fig_map.update_layout(
    paper_bgcolor="#050816",
    plot_bgcolor="#050816",
    geo=dict(
        bgcolor="#050816",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="rgba(0,229,255,0.15)",
        showland=True,
        landcolor="#0d1324",
        showocean=True,
        oceancolor="#050816",
        showlakes=False,
        showcountries=True,
        countrycolor="rgba(0,229,255,0.08)"
    ),
    font_color="white",
    title_x=0.5,
    title_font=dict(size=14, color="#00e5ff"),
    coloraxis_colorbar=dict(
        title=dict(
            text="Risk (log10)",
            font=dict(color="rgba(255,255,255,0.5)", size=10)
        ),
        tickfont=dict(color="rgba(255,255,255,0.5)", size=10),
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------

color_discrete = {
    "Düşük Risk":  "#39ff7a",
    "Orta Risk":   "#ffb800",
    "Yüksek Risk": "#ff2d55"
}

fig_bar = px.bar(
    map_df,
    x="country_name",
    y="risk_score",
    color="risk_level",
    color_discrete_map=color_discrete,
    labels={"country_name": "", "risk_score": "Risk Skoru", "risk_level": "Seviye"},
    text_auto=False
)

fig_bar.update_traces(marker_line_width=0, opacity=0.85)
fig_bar.update_layout(
    paper_bgcolor="#050816",
    plot_bgcolor="#0d1324",
    font_color="white",
    xaxis=dict(gridcolor="rgba(0,229,255,0.06)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(0,229,255,0.06)", tickfont=dict(size=10), tickformat=".2s"),
    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,229,255,0.15)"),
    margin=dict(l=0, r=0, t=10, b=0)
)

# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------

heatmap_data = pivot.drop(
    columns=["risk_score", "cluster", "anomaly", "ai_comment",
             "internet_users_m", "exposure_per_user"],
    errors="ignore"
)
heatmap_data = heatmap_data.select_dtypes(include=["number"])
heatmap_data.index = [country_names.get(x, x) for x in heatmap_data.index]

fig_heatmap, ax = plt.subplots(figsize=(13, 6))
fig_heatmap.patch.set_facecolor("#050816")
ax.set_facecolor("#050816")

sns.heatmap(
    np.log10(heatmap_data + 1),
    cmap="RdYlGn_r",
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    linecolor=(0.0, 0.898, 1.0, 0.08),
    ax=ax,
    cbar_kws={"shrink": 0.8}
)

plt.title("Exposure Heatmap (Log10)", color="#00e5ff", fontsize=13, pad=14)
plt.xticks(color=(1.0, 1.0, 1.0, 0.6), fontsize=9, rotation=30)
plt.yticks(color=(1.0, 1.0, 1.0, 0.7), fontsize=9)
ax.tick_params(colors=(1.0, 1.0, 1.0, 0.5))
plt.tight_layout()

# ---------------------------------------------------
# PORT KORELASYON MATRİSİ
# ---------------------------------------------------

corr_data = pivot.drop(
    columns=["risk_score", "cluster", "anomaly", "ai_comment",
             "internet_users_m", "exposure_per_user"],
    errors="ignore"
).select_dtypes(include=["number"])

corr_matrix = corr_data.corr()

fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
fig_corr.patch.set_facecolor("#050816")
ax_corr.set_facecolor("#050816")

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.4,
    linecolor=(0.0, 0.898, 1.0, 0.06),
    ax=ax_corr,
    cbar_kws={"shrink": 0.8},
    vmin=-1, vmax=1
)

plt.title("Port Korelasyon Matrisi", color="#00e5ff", fontsize=13, pad=14)
plt.xticks(color=(1.0, 1.0, 1.0, 0.7), fontsize=9, rotation=30)
plt.yticks(color=(1.0, 1.0, 1.0, 0.7), fontsize=9)
ax_corr.tick_params(colors=(1.0, 1.0, 1.0, 0.5))
plt.tight_layout()

# ---------------------------------------------------
# EXPOSURE PER USER CHART
# ---------------------------------------------------

exp_df = pivot.reset_index()[
    ["country", "risk_score", "internet_users_m", "exposure_per_user"]
].copy()
exp_df["country_name"] = exp_df["country"].map(country_names)
exp_df = exp_df.sort_values("exposure_per_user", ascending=False)

fig_exp = px.bar(
    exp_df,
    x="country_name",
    y="exposure_per_user",
    color="exposure_per_user",
    color_continuous_scale=["#39ff7a", "#ffb800", "#ff2d55"],
    labels={"country_name": "", "exposure_per_user": "Risk / İnternet Kullanıcısı"},
    title="🌐 Kişi Başına Normalize Edilmiş Exposure Oranı"
)

fig_exp.update_traces(marker_line_width=0)
fig_exp.update_layout(
    paper_bgcolor="#050816",
    plot_bgcolor="#0d1324",
    font_color="white",
    title_x=0.5,
    title_font=dict(color="#00e5ff", size=13),
    xaxis=dict(gridcolor="rgba(0,229,255,0.06)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(0,229,255,0.06)", tickfont=dict(size=10)),
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=50, b=0)
)

# ---------------------------------------------------
# ZAMAN SERİSİ CHART
# ---------------------------------------------------

if len(trend_df) > 0 and len(trend_df["timestamp"].unique()) > 1:
    trend_df["country_name"] = trend_df["country"].map(country_names)
    trend_df["ts_dt"] = pd.to_datetime(trend_df["timestamp"], format="%Y%m%d_%H%M%S")

    fig_trend = px.line(
        trend_df,
        x="ts_dt",
        y="risk_score",
        color="country_name",
        markers=True,
        labels={"ts_dt": "Tarih", "risk_score": "Risk Skoru", "country_name": "Ülke"},
        title="📈 Risk Skoru Zaman Serisi Trendi"
    )

    fig_trend.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#0d1324",
        font_color="white",
        title_x=0.5,
        title_font=dict(color="#00e5ff", size=13),
        xaxis=dict(gridcolor="rgba(0,229,255,0.06)"),
        yaxis=dict(gridcolor="rgba(0,229,255,0.06)", tickformat=".2s"),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=50, b=0)
    )
else:
    fig_trend = None

# ---------------------------------------------------
# CVE CHART
# ---------------------------------------------------

if not cve_df.empty:
    cve_df["cve_short"] = cve_df["cve"].str.extract(r"(CVE-[\d-]+)")
    cve_df["cve_short"] = cve_df["cve_short"].fillna(cve_df["cve"])

    fig_cve = px.bar(
        cve_df,
        x="cve_short",
        y="total",
        color="cve_short",
        color_discrete_sequence=["#ff6400", "#ff2d55", "#ffb800", "#ff9040"],
        labels={"cve_short": "CVE / Açık", "total": "Etkilenen Cihaz Sayısı"},
        title="🔓 Log4Shell & Kritik CVE Exposure"
    )

    fig_cve.update_traces(marker_line_width=0, opacity=0.9)
    fig_cve.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#0d1324",
        font_color="white",
        title_x=0.5,
        title_font=dict(color="#ff6400", size=13),
        showlegend=False,
        xaxis=dict(gridcolor="rgba(0,229,255,0.04)", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,229,255,0.04)", tickformat=".2s"),
        margin=dict(l=0, r=0, t=50, b=0)
    )

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📌 Proje Süreci",
    "📊 Risk Analizi",
    "🔬 Gelişmiş Analiz"
])

# ===================================================
# TAB 1 — PROJE SÜRECİ
# ===================================================

with tab1:

    st.header("📌 Proje Özeti")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Veri Kaynağı</h3>
            <h1>Shodan API</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card warn">
            <h3>Dil</h3>
            <h1>Python</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Veri Saklama Formatı</h3>
            <h1>CSV</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card ok">
            <h3>AI Modeli</h3>
            <h1 style="font-size:1.1rem;">Isolation Forest</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="panel">
        <div class="panel-title">▌ Proje Adımları</div>

        <div class="process-step">
            <div class="step-num">01</div>
            <div>
                <div class="step-title">Veri Toplama</div>
                <div class="step-item">Shodan API ile ülke bazlı servis taraması</div>
                <div class="step-item">port:22 · port:3389 · port:21 · port:445 sorguları</div>
                <div class="step-item">MongoDB · Redis · Elasticsearch exposure</div>
                <div class="step-item">Log4Shell & CVE bazlı güvenlik açığı sorguları</div>
                <div class="step-item">Ham veri zaman damgalı CSV olarak kaydedildi</div>
            </div>
        </div>

        <div class="process-step">
            <div class="step-num">02</div>
            <div>
                <div class="step-title">Veri Temizleme & Ön İşleme</div>
                <div class="step-item">Pivot table oluşturuldu (ülke × servis)</div>
                <div class="step-item">Eksik değerler fillna(0) ile dolduruldu</div>
                <div class="step-item">Log10 scaling uygulandı</div>
                <div class="step-item">StandardScaler ile normalizasyon</div>
                <div class="step-item">İnternet kullanıcı nüfusuna göre normalize oranı</div>
            </div>
        </div>

        <div class="process-step">
            <div class="step-num">03</div>
            <div>
                <div class="step-title">AI Analizi</div>
                <div class="step-item">K-Means (k=3): Düşük / Orta / Yüksek Risk kümesi</div>
                <div class="step-item">Isolation Forest: anomali tespiti (contamination=%20)</div>
                <div class="step-item">Servis bazlı ağırlıklı risk skoru hesabı</div>
                <div class="step-item">Port korelasyon matrisi analizi</div>
                <div class="step-item">Dinamik AI yorum sistemi</div>
            </div>
        </div>

        <div class="process-step" style="border-bottom:none;margin-bottom:0;padding-bottom:0;">
            <div class="step-num">04</div>
            <div>
                <div class="step-title">Görselleştirme</div>
                <div class="step-item">Streamlit + Plotly + Seaborn dashboard</div>
                <div class="step-item">Dünya koroplet haritası (siyah tema)</div>
                <div class="step-item">Zaman serisi risk trendi</div>
                <div class="step-item">Kişi başına normalize exposure oranı</div>
                <div class="step-item">CVE / Log4Shell exposure grafiği</div>
            </div>
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col_right:

        st.markdown("""
        <div class="panel">
        <div class="panel-title">▌ Risk Ağırlık Matrisi</div>
        """, unsafe_allow_html=True)

        weight_df = pd.DataFrame({
            "Servis":  ["RDP", "SMB", "FTP", "MongoDB", "Redis", "SSH", "Elasticsearch"],
            "Port":    ["3389", "445", "21", "27017", "6379", "22", "9200"],
            "Ağırlık": [5, 5, 4, 4, 4, 3, 3],
            "Tehdit":  [
                "Uzak masaüstü brute-force",
                "EternalBlue / ransomware",
                "Açık metin kimlik bilgisi",
                "Auth-bypass DB exposure",
                "Yetkisiz veri erişimi",
                "Kaba kuvvet saldırısı",
                "İndeks veri sızıntısı"
            ]
        })

        st.dataframe(weight_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="panel-title" style="margin-top:16px;">▌ Kaydedilen Dosyalar</div>
        <div class="ai-text" style="font-family:'Share Tech Mono',monospace; font-size:0.78rem; line-height:2;">
            📁 {csv_dir}/{os.path.basename(raw_csv_path)}<br>
            📁 {csv_dir}/{os.path.basename(risk_csv_path)}<br>
            📁 {csv_dir}/{os.path.basename(ai_csv_path)}<br>
        </div>

        <div class="panel-title" style="margin-top:16px;">▌ Teknoloji Yığını</div>
        <div>
            <span class="chip">Python 3.11</span>
            <span class="chip">Shodan API</span>
            <span class="chip">Pandas</span>
            <span class="chip">Scikit-learn</span>
            <span class="chip">Plotly</span>
            <span class="chip">Streamlit</span>
            <span class="chip">Seaborn</span>
            <span class="chip">Isolation Forest</span>
            <span class="chip">K-Means</span>
            <span class="chip">StandardScaler</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ===================================================
# TAB 2 — RİSK ANALİZİ
# ===================================================

with tab2:

    st.header("📊 Risk Analizi")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Analiz Edilen Ülke</h3>
            <h1>{len(pivot.index)}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_exp = int(filtered_df["total"].sum())
        st.markdown(f"""
        <div class="metric-card warn">
            <h3>Toplam Exposure</h3>
            <h1>{total_exp:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        top_country = country_names.get(pivot["risk_score"].idxmax(), "?")
        st.markdown(f"""
        <div class="metric-card danger">
            <h3>En Riskli Ülke</h3>
            <h1>{top_country}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        sus_count = (pivot["anomaly_label"] == "Şüpheli").sum()
        st.markdown(f"""
        <div class="metric-card {'danger' if sus_count > 0 else 'ok'}">
            <h3>Şüpheli Anomali</h3>
            <h1>{sus_count}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- TÜRKİYE ÖZEL ANALİZ ---

    st.subheader("🇹🇷 Türkiye Özel Risk Analizi")

    if "TR" in pivot.index:

        tr_score    = int(pivot.loc["TR", "risk_score"])
        tr_level    = pivot.loc["TR", "risk_level"]
        tr_comment  = pivot.loc["TR", "ai_comment"]
        tr_anomaly  = pivot.loc["TR", "anomaly_label"]
        tr_exp_rate = pivot.loc["TR", "exposure_per_user"]

        level_class = (
            "danger" if "Yüksek" in tr_level else
            "warn"   if "Orta"   in tr_level else
            "ok"
        )

        col_tr1, col_tr2 = st.columns([1, 2])

        with col_tr1:
            st.markdown(f"""
            <div class="metric-card {level_class}" style="text-align:left; padding:20px;">
                <h2 style="font-size:1.6rem;">🇹🇷 Türkiye</h2>
                <br>
                <h3>Risk Seviyesi</h3>
                <h1 style="font-size:1.3rem;">{tr_level}</h1>
                <br>
                <h3>Risk Skoru</h3>
                <h1>{tr_score:,}</h1>
                <br>
                <h3>Kişi Başına Oran</h3>
                <h1 style="font-size:1.1rem;">{tr_exp_rate:.4f}</h1>
                <br>
                <h3>Anomali Durumu</h3>
                <h1 style="font-size:1rem;">{tr_anomaly}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col_tr2:
            services = {}
            for key, label in [
                ("port:22",   "SSH"),
                ("port:3389", "RDP"),
                ("port:21",   "FTP"),
                ("port:445",  "SMB")
            ]:
                if key in pivot.columns:
                    services[label] = pivot.loc["TR", key]

            if services:
                pie_df = pd.DataFrame({
                    "Servis":   list(services.keys()),
                    "Exposure": list(services.values())
                })

                pie_fig = px.pie(
                    pie_df,
                    names="Servis",
                    values="Exposure",
                    title="Türkiye Port Exposure Dağılımı",
                    color_discrete_sequence=["#00e5ff", "#ff2d55", "#ffb800", "#39ff7a"],
                    hole=0.4
                )

                pie_fig.update_traces(
                    textfont_size=11,
                    marker=dict(line=dict(color="#050816", width=2))
                )

                pie_fig.update_layout(
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816",
                    font_color="white",
                    title_x=0.5,
                    title_font=dict(color="#00e5ff", size=13),
                    legend=dict(font=dict(size=11)),
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown(f"""
            <div class="ai-block {level_class.replace('danger','high').replace('warn','med').replace('ok','low')}">
                <div class="ai-country {'med' if 'Orta' in tr_level else 'high' if 'Yüksek' in tr_level else ''}">
                    🇹🇷 Türkiye — AI Risk Yorumu
                </div>
                <div class="ai-text">{tr_comment}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🌍 Dünya Risk Haritası")
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Ülke Risk Skorları")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    st.subheader("🔥 Exposure Heatmap")
    st.pyplot(fig_heatmap)

    st.markdown("---")

    st.subheader("🚨 Anomali Tespiti")

    anomaly_df = pivot.reset_index().copy()
    anomaly_df["Ülke"] = anomaly_df["country"].map(country_names)

    display_df = anomaly_df[
        ["Ülke", "risk_score", "risk_level", "anomaly_label", "exposure_per_user"]
    ].rename(columns={
        "risk_score":        "Risk Skoru",
        "risk_level":        "Risk Seviyesi",
        "anomaly_label":     "Anomali Durumu",
        "exposure_per_user": "Kişi Başına Oran"
    }).sort_values("Risk Skoru", ascending=False)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("🤖 AI Yorumları")

    for country in pivot.index:
        level   = pivot.loc[country, "risk_level"]
        anomaly = pivot.loc[country, "anomaly_label"]
        score   = int(pivot.loc[country, "risk_score"])
        comment = pivot.loc[country, "ai_comment"]

        block_class = (
            "high" if "Yüksek" in level else
            "med"  if "Orta"   in level else
            "low"
        )
        level_badge = (
            '<span class="badge badge-high">Yüksek Risk</span>' if "Yüksek" in level else
            '<span class="badge badge-med">Orta Risk</span>'    if "Orta"   in level else
            '<span class="badge badge-low">Düşük Risk</span>'
        )
        anomaly_badge = (
            '<span class="badge badge-sus">Şüpheli</span>'
            if anomaly == "Şüpheli"
            else '<span class="badge badge-normal">Normal</span>'
        )

        st.markdown(f"""
        <div class="ai-block {block_class}">
            <div class="ai-country {block_class}">
                🌐 {country_names.get(country, country)}
            </div>
            <div style="margin:6px 0; display:flex; gap:8px; flex-wrap:wrap;">
                {level_badge}
                {anomaly_badge}
                <span style="font-size:0.75rem; color:rgba(255,255,255,0.4);
                      font-family:'Share Tech Mono',monospace; align-self:center;">
                    score: {score:,}
                </span>
            </div>
            <div class="ai-text">{comment}</div>
        </div>
        """, unsafe_allow_html=True)

# ===================================================
# TAB 3 — GELİŞMİŞ ANALİZ
# ===================================================

with tab3:

    st.header("🔬 Gelişmiş Analiz")

    # --- BÖLÜM 1: ÜLKE BAŞINA EXPOSURE ORANI ---

    st.subheader("🌐 Kişi Başına Normalize Edilmiş Exposure Oranı")

    st.markdown("""
    <div class="ai-block" style="margin-bottom:12px;">
        <div class="ai-text">
            Ham risk skoru büyük nüfuslu ülkeleri yapısal olarak öne çıkarır.
            Bu grafik, risk skorunu o ülkenin internet kullanıcı sayısına bölerek
            <strong style="color:#00e5ff;">gerçek kişi başına tehdit yoğunluğunu</strong> gösterir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig_exp, use_container_width=True)

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        top_exp = exp_df.iloc[0]
        st.markdown(f"""
        <div class="metric-card danger">
            <h3>En Yüksek Kişi Başına Risk</h3>
            <h1 style="font-size:1.3rem;">{country_names.get(top_exp['country'], top_exp['country'])}</h1>
            <h3>Oran: {top_exp['exposure_per_user']:.4f}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col_exp2:
        bottom_exp = exp_df.iloc[-1]
        st.markdown(f"""
        <div class="metric-card ok">
            <h3>En Düşük Kişi Başına Risk</h3>
            <h1 style="font-size:1.3rem;">{country_names.get(bottom_exp['country'], bottom_exp['country'])}</h1>
            <h3>Oran: {bottom_exp['exposure_per_user']:.4f}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.dataframe(
        exp_df[["country_name", "internet_users_m", "risk_score", "exposure_per_user"]]
        .rename(columns={
            "country_name":      "Ülke",
            "internet_users_m":  "İnternet Kul. (M)",
            "risk_score":        "Risk Skoru",
            "exposure_per_user": "Kişi Başına Oran"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --- BÖLÜM 2: PORT KORELASYON MATRİSİ ---

    st.subheader("🔗 Port Korelasyon Matrisi")

    st.markdown("""
    <div class="ai-block" style="margin-bottom:12px;">
        <div class="ai-text">
            Hangi portlar birlikte açık kalıyor? Yüksek korelasyon, aynı altyapının
            birden fazla servis açığına sahip olduğuna işaret eder — bu da saldırı
            yüzeyi genişliğinin bir göstergesidir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.pyplot(fig_corr)

    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                "Port A":      corr_matrix.columns[i],
                "Port B":      corr_matrix.columns[j],
                "Korelasyon":  round(corr_matrix.iloc[i, j], 3)
            })

    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values(
        "Korelasyon", ascending=False, key=abs
    )

    st.markdown("**En Güçlü Port Korelasyonları:**")
    st.dataframe(
        corr_pairs_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --- BÖLÜM 3: LOG4SHELL / CVE SORGUSU ---

    st.subheader("🔓 Log4Shell & Kritik CVE Exposure")

    st.markdown("""
    <div class="ai-block cve" style="margin-bottom:12px;">
        <div class="ai-country cve-title">⚠️ CVE Bazlı Tehdit Analizi</div>
        <div class="ai-text">
            Shodan'ın <code>vuln:</code> filtresi ile kritik güvenlik açıklarına sahip
            cihazlar sorgulanmıştır. Log4Shell (CVE-2021-44228), dünya genelinde milyonlarca
            sistemi etkilemiş en kritik RCE açıklarından biridir.
            <br><br>
            Veri <code>shodan_fetch.py</code> ile çekilip <code>cve_data.csv</code> olarak
            kaydedilmiştir. Aşağıdaki sorgular kullanılmıştır:
        </div>
    </div>
    """, unsafe_allow_html=True)

    for cve_name, query in cve_queries.items():
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
             color:#ff6400; padding:3px 0;">
            › {cve_name} &nbsp;|&nbsp;
            <span style="color:rgba(255,255,255,0.4);">shodan query: {query}</span>
        </div>
        """, unsafe_allow_html=True)

    st.plotly_chart(fig_cve, use_container_width=True)
    st.dataframe(cve_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- BÖLÜM 4: ZAMAN SERİSİ ---

    st.subheader("📈 Risk Skoru Zaman Serisi Trendi")

    if fig_trend is not None:
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.markdown(f"""
        <div class="ai-block" style="margin-bottom:0;">
            <div class="ai-country">⏳ Yeterli Tarihsel Veri Yok</div>
            <div class="ai-text">
                Zaman serisi grafiği için en az 2 farklı tarihe ait veri gerekmektedir.<br>
                Dashboard her çalıştırıldığında <code>data_history/</code> klasörüne
                zaman damgalı CSV kaydedilir. Birkaç çalıştırma sonrasında trend grafiği
                burada otomatik olarak görünecektir.<br><br>
                Şu ana kadar kaydedilen snapshot sayısı:
                <strong style="color:#00e5ff;">{len(history_files)}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- BÖLÜM 5: CVE RİSK ANALİZİ ---

    st.subheader("🧬 CVE Bazlı Derinlemesine Risk Analizi")

    if not cve_df.empty and "total" in cve_df.columns:

        total_cve_devices = int(cve_df["total"].sum())
        max_cve_row       = cve_df.loc[cve_df["total"].idxmax()]
        min_cve_row       = cve_df.loc[cve_df["total"].idxmin()]

        # Özet metrik kartları
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown(f"""
            <div class="metric-card danger">
                <h3>Toplam CVE Açık Cihaz</h3>
                <h1>{total_cve_devices:,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown(f"""
            <div class="metric-card danger">
                <h3>En Yaygın Açık</h3>
                <h1 style="font-size:0.95rem;">{max_cve_row['cve_short'] if 'cve_short' in cve_df.columns else max_cve_row['cve']}</h1>
                <h3>{int(max_cve_row['total']):,} cihaz</h3>
            </div>
            """, unsafe_allow_html=True)

        with col_c3:
            st.markdown(f"""
            <div class="metric-card warn">
                <h3>En Az Yaygın Açık</h3>
                <h1 style="font-size:0.95rem;">{min_cve_row['cve_short'] if 'cve_short' in cve_df.columns else min_cve_row['cve']}</h1>
                <h3>{int(min_cve_row['total']):,} cihaz</h3>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CVE yüzde dağılımı + AI yorum
        col_pie, col_ai = st.columns([1, 1])

        with col_pie:
            cve_label_col = "cve_short" if "cve_short" in cve_df.columns else "cve"
            fig_cve_pie = px.pie(
                cve_df,
                names=cve_label_col,
                values="total",
                title="CVE Exposure Dağılımı (%)",
                color_discrete_sequence=["#ff6400", "#ff2d55", "#ffb800", "#ff9040"],
                hole=0.45
            )
            fig_cve_pie.update_traces(
                textfont_size=11,
                marker=dict(line=dict(color="#050816", width=2))
            )
            fig_cve_pie.update_layout(
                paper_bgcolor="#050816",
                plot_bgcolor="#050816",
                font_color="white",
                title_x=0.5,
                title_font=dict(color="#ff6400", size=13),
                legend=dict(font=dict(size=10)),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig_cve_pie, use_container_width=True)

        with col_ai:
            # Her CVE için tehdit yorumu
            cve_comments = {
                "CVE-2021-44228": {
                    "title": "Log4Shell",
                    "severity": "KRİTİK",
                    "cls": "high",
                    "text": (
                        "Java Log4j kütüphanesinde uzaktan kod çalıştırma (RCE) açığı. "
                        "Kimlik doğrulaması gerektirmez; tek bir log satırı ile sistem "
                        "ele geçirilebilir. Bulut, finans ve e-ticaret altyapılarını doğrudan hedef alır."
                    )
                },
                "CVE-2021-26855": {
                    "title": "ProxyLogon",
                    "severity": "KRİTİK",
                    "cls": "high",
                    "text": (
                        "Microsoft Exchange Server SSRF açığı. Kimlik doğrulaması atlanarak "
                        "posta sunucusuna tam erişim sağlanabilir. Kurumsal e-posta altyapıları "
                        "birincil hedef konumundadır."
                    )
                },
                "CVE-2017-0144": {
                    "title": "EternalBlue",
                    "severity": "YÜKSEK",
                    "cls": "high",
                    "text": (
                        "SMBv1 protokolündeki buffer overflow açığı. WannaCry ve NotPetya "
                        "ransomware saldırılarında kullanılmıştır. Yamasız Windows sistemler "
                        "hâlâ geniş çaplı risk altındadır."
                    )
                },
                "CVE-2014-0160": {
                    "title": "Heartbleed",
                    "severity": "YÜKSEK",
                    "cls": "med",
                    "text": (
                        "OpenSSL TLS heartbeat uzantısındaki bellek sızıntısı açığı. "
                        "Sunucu belleğinden özel anahtarlar, şifreler ve oturum verileri "
                        "okunabilir. Eski sürüm OpenSSL kullanan sistemlerde hâlâ aktif risk."
                    )
                },
            }

            st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
            for _, row in cve_df.iterrows():
                cve_id  = row.get("cve_short", row["cve"])
                comment = next(
                    (v for k, v in cve_comments.items() if k in row["cve"]),
                    {"title": cve_id, "severity": "BİLİNMİYOR", "cls": "cve",
                     "text": "Detaylı analiz mevcut değil."}
                )
                pct = round(row["total"] / total_cve_devices * 100, 1)
                st.markdown(f"""
                <div class="ai-block {comment['cls']}" style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span class="ai-country cve-title" style="font-size:0.9rem;">
                            🔴 {comment['title']} — {cve_id}
                        </span>
                        <span>
                            <span class="badge badge-cve">{comment['severity']}</span>
                            &nbsp;
                            <span style="font-family:'Share Tech Mono',monospace;
                                  font-size:0.75rem; color:#ff6400;">
                                {int(row['total']):,} cihaz &nbsp;({pct}%)
                            </span>
                        </span>
                    </div>
                    <div class="ai-text">{comment['text']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # CVE karşılaştırmalı yatay bar
        st.markdown("<br>", unsafe_allow_html=True)
        cve_sorted = cve_df.sort_values("total", ascending=True)
        cve_label_col = "cve_short" if "cve_short" in cve_df.columns else "cve"

        fig_cve_h = px.bar(
            cve_sorted,
            x="total",
            y=cve_label_col,
            orientation="h",
            color="total",
            color_continuous_scale=["#ffb800", "#ff6400", "#ff2d55"],
            labels={cve_label_col: "", "total": "Etkilenen Cihaz Sayısı"},
            title="🔎 CVE Karşılaştırmalı Exposure (Büyükten Küçüğe)"
        )
        fig_cve_h.update_traces(marker_line_width=0, opacity=0.9)
        fig_cve_h.update_layout(
            paper_bgcolor="#050816",
            plot_bgcolor="#0d1324",
            font_color="white",
            title_x=0.5,
            title_font=dict(color="#ff6400", size=13),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(0,229,255,0.06)", tickformat=".2s"),
            yaxis=dict(gridcolor="rgba(0,229,255,0.04)", tickfont=dict(size=11)),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig_cve_h, use_container_width=True)

        # Ham tablo
        st.markdown("**📋 CVE Ham Veri Tablosu:**")
        st.dataframe(
            cve_df.drop(columns=["cve_short"], errors="ignore")
                  .rename(columns={"cve": "CVE Adı", "query": "Shodan Sorgusu", "total": "Etkilenen Cihaz"}),
            use_container_width=True,
            hide_index=True
        )

    # --- BÖLÜM 6: HAM VERİ ÖNCEL ---

    st.markdown("---")
    st.subheader("📋 Ham Veri Önizlemesi")

    st.markdown("""
    <div class="ai-block" style="margin-bottom:12px;">
        <div class="ai-text">
            <code>raw_data.csv</code> — Shodan API'den çekilen ve analiz sürecinde
            kullanılan ham exposure verisi.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.head(30), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.78rem;
         color:rgba(255,255,255,0.35); text-align:right; margin-top:6px;">
        toplam {len(df)} satır · {len(df.columns)} sütun
    </div>
    """, unsafe_allow_html=True)