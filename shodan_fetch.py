import shodan
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SHODAN_API_KEY")
api = shodan.Shodan(API_KEY)

cve_queries = {
    "Log4Shell (CVE-2021-44228)":  "vuln:CVE-2021-44228",
    "ProxyLogon (CVE-2021-26855)": "vuln:CVE-2021-26855",
    "EternalBlue (CVE-2017-0144)": "vuln:CVE-2017-0144",
    "Heartbleed (CVE-2014-0160)":  "vuln:CVE-2014-0160",
}

records = []

for cve_name, query in cve_queries.items():
    try:
        result = api.count(query)
        records.append({
            "cve":   cve_name,
            "query": query,
            "total": result["total"]
        })
        print(f"{cve_name}: {result['total']:,} cihaz")
    except shodan.APIError as e:
        print(f"Hata ({cve_name}): {e}")

df = pd.DataFrame(records)
df.to_csv("cve_data.csv", index=False)
print("\nKaydedildi: cve_data.csv")