import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------
# CSV OKU
# ---------------------------------------------------

df = pd.read_csv("raw_data.csv")

print("\nHAM VERİ:\n")
print(df.head())

# ---------------------------------------------------
# VERİ TEMİZLEME
# ---------------------------------------------------

df = df.dropna(subset=["total"])            # boş satırları sil
df = df[df["total"] >= 0]                   # negatif değerleri kaldır
df["service"] = df["service"].str.strip()   # boşluk temizle
df["country"] = df["country"].str.strip()
df["total"] = df["total"].astype(int)

print("\nTEMİZLENMİŞ VERİ:\n")
print(df.head())

# ---------------------------------------------------
# PIVOT TABLO
# ---------------------------------------------------

pivot = df.pivot(
    index="country",
    columns="service",
    values="total"
)

print("\nPIVOT TABLO:\n")
print(pivot)

# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------

plt.figure(figsize=(12, 6))

sns.heatmap(
    np.log10(pivot + 1),
    annot=True,
    cmap="Reds"
)

plt.title("Log-Scaled Global Exposure Heatmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# RISK MODELİ
# ---------------------------------------------------

risk_weights = {
    "port:22": 3,
    "port:3389": 5,
    "port:21": 4,
    "port:445": 5,
    "mongodb": 4,
    "redis": 4,
    "elasticsearch": 3
}

df["weight"] = (
    df["service"]
    .map(risk_weights)
    .fillna(1)
)

df["risk_score"] = df["total"] * df["weight"]

# ---------------------------------------------------
# ÜLKE BAZLI RİSK
# ---------------------------------------------------

country_risk = (
    df.groupby("country")["risk_score"]
    .sum()
    .sort_values(ascending=False)
)

print("\nÜLKE RİSK SKORLARI:\n")
print(country_risk)

# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------

plt.figure(figsize=(10, 5))

country_risk.plot(kind="bar")

plt.title("Cyber Risk Index by Country")
plt.ylabel("Risk Score")

plt.tight_layout()
plt.show()