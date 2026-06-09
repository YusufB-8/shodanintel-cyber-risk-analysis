import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# ---------------------------------------------------
# ENV YÜKLE
# ---------------------------------------------------

load_dotenv()

API_KEY = os.getenv("SHODAN_API_KEY")

if not API_KEY:
    raise ValueError("SHODAN_API_KEY bulunamadı!")

# ---------------------------------------------------
# ÜLKELER
# ---------------------------------------------------

countries = [
    "TR",  # Türkiye
    "US",  # ABD
    "DE",  # Almanya
    "FR",  # Fransa
    "RU",  # Rusya
    "CN",  # Çin
    "GB",  # İngiltere
    "NL",  # Hollanda
    "JP",  # Japonya
    "IN",  # Hindistan
    "BR"   # Brezilya
]

# ---------------------------------------------------
# SERVİSLER
# ---------------------------------------------------

services = [
    "port:22",          # SSH
    "port:3389",        # RDP
    "port:21",          # FTP
    "port:445",         # SMB
    "mongodb",
    "redis",
    "elasticsearch"
]

# ---------------------------------------------------
# API
# ---------------------------------------------------

BASE_URL = "https://api.shodan.io/shodan/host/count"

results = []

# ---------------------------------------------------
# VERİ TOPLAMA
# ---------------------------------------------------

for country in countries:

    print(f"\n[ÜLKE] {country}")

    for service in services:

        query = f"{service} country:{country}"

        params = {
            "key": API_KEY,
            "query": query
        }

        try:
            response = requests.get(
                BASE_URL,
                params=params,
                timeout=15
            )

            response.raise_for_status()

            data = response.json()

            total = data.get("total", 0)

            results.append({
                "country": country,
                "service": service,
                "query": query,
                "total": total
            })

            print(f"[OK] {query:<35} -> {total}")

        except requests.exceptions.RequestException as e:

            print(f"[HATA] {query}")
            print(e)

        # Rate limit dostu bekleme
        time.sleep(1.2)

# ---------------------------------------------------
# DATAFRAME
# ---------------------------------------------------

df = pd.DataFrame(results)

# Büyükten küçüğe sırala
df = df.sort_values(
    by="total",
    ascending=False
)

# ---------------------------------------------------
# CSV KAYDET
# ---------------------------------------------------

output_file = "raw_data.csv"

df.to_csv(output_file, index=False)

print(f"\nVeri başarıyla kaydedildi: {output_file}")

# ---------------------------------------------------
# ÖZET
# ---------------------------------------------------

print("\nİlk 10 kayıt:\n")
print(df.head(10))