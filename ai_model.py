import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# VERİYİ OKU
# ---------------------------------------------------

print("\n[1] Veri okunuyor...")

df = pd.read_csv("raw_data.csv")

# ---------------------------------------------------
# PIVOT TABLO
# ---------------------------------------------------

print("[2] Pivot tablo oluşturuluyor...")

pivot = df.pivot(
    index="country",
    columns="service",
    values="total"
).fillna(0)

print("\nPivot Tablo:\n")
print(pivot)

# ---------------------------------------------------
# RISK SCORE
# ---------------------------------------------------

print("\n[3] Risk skorları hesaplanıyor...")

risk_weights = {
    "port:22": 3,
    "port:3389": 5,
    "port:21": 4,
    "port:445": 5,
    "mongodb": 4,
    "redis": 4,
    "elasticsearch": 3
}

# ağırlık sütunu
pivot["risk_score"] = 0

for service, weight in risk_weights.items():
    if service in pivot.columns:
        pivot["risk_score"] += pivot[service] * weight

# ---------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------

print("[4] Normalization yapılıyor...")

features = pivot.drop(columns=["risk_score"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ---------------------------------------------------
# AI MODELİ
# ---------------------------------------------------

print("[5] KMeans modeli çalıştırılıyor...")

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

pivot["cluster"] = model.fit_predict(X_scaled)

# ---------------------------------------------------
# CLUSTER -> RISK LABEL
# ---------------------------------------------------

cluster_risk = pivot.groupby("cluster")["risk_score"].mean()

sorted_clusters = cluster_risk.sort_values().index.tolist()

cluster_map = {
    sorted_clusters[0]: "Low Risk",
    sorted_clusters[1]: "Medium Risk",
    sorted_clusters[2]: "High Risk"
}

pivot["risk_level"] = pivot["cluster"].map(cluster_map)

# ---------------------------------------------------
# AI YORUM FONKSİYONU
# ---------------------------------------------------

def generate_comment(row):

    level = row["risk_level"]
    score = row["risk_score"]

    ssh = row.get("port:22", 0)
    rdp = row.get("port:3389", 0)
    ftp = row.get("port:21", 0)
    smb = row.get("port:445", 0)
    mongo = row.get("mongodb", 0)
    redis = row.get("redis", 0)
    elastic = row.get("elasticsearch", 0)

    reasons = []
    attacks = []

    # SSH
    if ssh > 500000:
        reasons.append(
            "yüksek SSH exposure"
        )
        attacks.append(
            "brute-force saldırıları"
        )

    # RDP
    if rdp > 100000:
        reasons.append(
            "yüksek RDP exposure"
        )
        attacks.append(
            "ransomware ve credential attack"
        )

    # FTP
    if ftp > 100000:
        reasons.append(
            "yüksek FTP exposure"
        )
        attacks.append(
            "zayıf kimlik doğrulama saldırıları"
        )

    # SMB
    if smb > 30000:
        reasons.append(
            "SMB servis exposure"
        )
        attacks.append(
            "worm propagation ve lateral movement"
        )

    # MongoDB
    if mongo > 5000:
        reasons.append(
            "yüksek MongoDB exposure"
        )
        attacks.append(
            "veri sızıntısı"
        )

    # Redis
    if redis > 5000:
        reasons.append(
            "yüksek Redis exposure"
        )
        attacks.append(
            "unauthorized cache access"
        )

    # Elasticsearch
    if elastic > 2000:
        reasons.append(
            "yüksek Elasticsearch exposure"
        )
        attacks.append(
            "search database leakage"
        )

    if not reasons:
        reasons.append(
            "orta seviyede exposure"
        )

    if not attacks:
        attacks.append(
            "genel servis saldırıları"
        )

    comment = (
        f"Risk Seviyesi: {level}. "
        f"Toplam risk skoru: {int(score)}. "
        f"Bu ülke {' , '.join(reasons)} nedeniyle riskli görünmektedir. "
        f"Potansiyel tehditler: {' , '.join(attacks)}."
    )

    return comment

# ---------------------------------------------------
# AI YORUM ÜRET
# ---------------------------------------------------

print("[6] AI yorumları oluşturuluyor...")

pivot["ai_comment"] = pivot.apply(
    generate_comment,
    axis=1
)

# ---------------------------------------------------
# SONUÇLAR
# ---------------------------------------------------

print("\nAI SONUÇLARI:\n")

for country in pivot.index:

    print("=" * 70)
    print(f"Ülke: {country}")
    print(f"Risk Seviyesi: {pivot.loc[country, 'risk_level']}")
    print(f"Risk Skoru: {int(pivot.loc[country, 'risk_score'])}")
    print(f"AI Yorumu: {pivot.loc[country, 'ai_comment']}")
    print("=" * 70)

# ---------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------

pivot.to_csv(
    "ai_risk_report.csv"
)

print("\nai_risk_report.csv oluşturuldu.")

# ---------------------------------------------------
# GÖRSEL 1 -> RISK BAR CHART
# ---------------------------------------------------

print("[7] Grafikler oluşturuluyor...")

plt.figure(figsize=(12, 6))

sns.barplot(
    x=pivot.index,
    y=pivot["risk_score"]
)

plt.title("Global Cyber Risk Scores")
plt.xlabel("Country")
plt.ylabel("Risk Score")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# GÖRSEL 2 -> HEATMAP
# ---------------------------------------------------

plt.figure(figsize=(12, 6))

heatmap_data = pivot.drop(
    columns=[
        "risk_score",
        "cluster",
        "risk_level",
        "ai_comment"
    ]
)

sns.heatmap(
    np.log10(heatmap_data + 1),
    cmap="Reds",
    annot=True
)

plt.title("Log Scaled Exposure Heatmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# GÖRSEL 3 -> AI CLUSTER
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    pivot["port:22"],
    pivot["port:3389"],
    c=pivot["cluster"],
    s=200
)

for country in pivot.index:
    plt.text(
        pivot.loc[country, "port:22"],
        pivot.loc[country, "port:3389"],
        country,
        fontsize=10
    )

plt.xlabel("SSH Exposure")
plt.ylabel("RDP Exposure")

plt.title("AI-Based Country Risk Clustering")

plt.tight_layout()
plt.show()

print("\nAnaliz tamamlandı.")