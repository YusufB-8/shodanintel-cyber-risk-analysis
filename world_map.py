import pandas as pd
import plotly.express as px
import numpy as np

# ---------------------------------------------------
# CSV OKU
# ---------------------------------------------------

df = pd.read_csv("raw_data.csv")

# ---------------------------------------------------
# ISO2 -> ISO3
# ---------------------------------------------------

country_map = {
    "TR": "TUR",
    "US": "USA",
    "DE": "DEU",
    "FR": "FRA",
    "RU": "RUS",
    "CN": "CHN",
    "GB": "GBR",
    "NL": "NLD",
    "JP": "JPN",
    "IN": "IND",
    "BR": "BRA"
}

df["country_iso3"] = df["country"].map(country_map)

# ---------------------------------------------------
# RISK WEIGHTS
# ---------------------------------------------------

weights = {
    "port:22": 3,
    "port:3389": 5,
    "port:21": 4,
    "port:445": 5,
    "mongodb": 4,
    "redis": 4,
    "elasticsearch": 3
}

df["weight"] = df["service"].map(weights)

# ---------------------------------------------------
# RISK HESABI
# ---------------------------------------------------

df["risk"] = df["total"] * df["weight"]

risk_country = (
    df.groupby("country_iso3")["risk"]
    .sum()
    .reset_index()
)

# ---------------------------------------------------
# LOG SCALE
# ---------------------------------------------------

risk_country["risk_log"] = np.log10(
    risk_country["risk"] + 1
)

print(risk_country)

# ---------------------------------------------------
# HARITA
# ---------------------------------------------------

fig = px.choropleth(
    risk_country,
    locations="country_iso3",
    color="risk_log",
    locationmode="ISO-3",
    color_continuous_scale="Reds",
    title="🌍 Global Cyber Risk Map"
)

fig.update_layout(
    title_x=0.5,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth"
    )
)

fig.show()