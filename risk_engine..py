import pandas as pd

# ---------------------------------------------------
# CSV OKU
# ---------------------------------------------------

df = pd.read_csv("raw_data.csv")

# ---------------------------------------------------
# RISK AĞIRLIKLARI
# ---------------------------------------------------

risk_weights = {
    "port:22": 3,          # SSH
    "port:3389": 5,        # RDP
    "port:21": 4,          # FTP
    "port:445": 5,         # SMB
    "mongodb": 4,
    "redis": 4,
    "elasticsearch": 3
}

# ---------------------------------------------------
# RISK SCORE
# ---------------------------------------------------

df["risk_weight"] = df["service"].map(risk_weights)

df["risk_score"] = (
    df["total"] * df["risk_weight"]
)

# ---------------------------------------------------
# ÜLKE BAZLI TOPLAM RISK
# ---------------------------------------------------

country_risk = (
    df.groupby("country")["risk_score"]
    .sum()
    .sort_values(ascending=False)
)

# ---------------------------------------------------
# SONUÇ
# ---------------------------------------------------

print("\nÜLKE RİSK SKORLARI:\n")
print(country_risk)

# ---------------------------------------------------
# CSV KAYDET
# ---------------------------------------------------

country_risk.to_csv("risk_scores.csv")

print("\nrisk_scores.csv oluşturuldu.")