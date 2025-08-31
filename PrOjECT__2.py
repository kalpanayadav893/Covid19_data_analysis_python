# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CSV_PATH = r"C:\Users\kalpa\Downloads\country_wise_latest.csv"
OUT_DIR = Path("figures")  # charts + KPIs will be saved here
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Normalize column names 
df.columns = (df.columns.str.strip()
                         .str.replace("/", "_", regex=False)
                         .str.replace(" ", "_", regex=False))

def pick(*candidates):
    cset = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower().replace("/", "").replace(" ", "")
        if key in cset:
            return cset[key]
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")

COL_COUNTRY  = pick("Country_Region", "Country/Region")
COL_CONF     = pick("Confirmed")
COL_DEATHS   = pick("Deaths")
COL_REC      = pick("Recovered")
COL_ACTIVE   = pick("Active")
COL_REGION   = pick("WHO_Region", "WHO Region") if any(
    c.lower().replace(" ", "_") in {"who_region"} for c in df.columns
) else None

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

df["CFR_%"] = np.where(df[COL_CONF] > 0, df[COL_DEATHS] / df[COL_CONF] * 100, 0)
df["Recovery_Rate_%"] = np.where(df[COL_CONF] > 0, df[COL_REC] / df[COL_CONF] * 100, 0)

# Global KPIs 
kpis = {
    "countries": int(df.shape[0]),
    "total_confirmed": int(df[COL_CONF].sum()),
    "total_deaths": int(df[COL_DEATHS].sum()),
    "total_recovered": int(df[COL_REC].sum()),
    "global_cfr_%": round((df[COL_DEATHS].sum() / df[COL_CONF].sum()) * 100, 2) if df[COL_CONF].sum() else 0,
    "avg_recovery_rate_%": round(df["Recovery_Rate_%"].mean(), 2),
}
pd.Series(kpis).to_csv(OUT_DIR / "summary_kpis.csv")
print("KPIs saved to", OUT_DIR / "summary_kpis.csv")

print("\nColumns:", df.columns.tolist())
print("\nDescribe (numeric):\n", df[[COL_CONF, COL_DEATHS, COL_REC, COL_ACTIVE]].describe())
print("\nNulls:\n", df.isnull().sum())

def save_show(name):
    plt.tight_layout()
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print("Saved:", path)

# Top 10 charts
for metric, title in [
    (COL_CONF,  "top10_confirmed"),
    (COL_DEATHS,"top10_deaths"),
    (COL_REC,   "top10_recovered"),
]:
    top = df.nlargest(10, metric)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y=COL_COUNTRY, data=top)
    plt.title(title.replace("_", " ").title())
    save_show(title)

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[[COL_CONF, COL_DEATHS, COL_REC, COL_ACTIVE]].corr(), annot=True, fmt=".2f")
plt.title("Correlation among Confirmed, Deaths, Recovered, Active")
save_show("correlation_heatmap")

# WHO Region aggregation 
if COL_REGION:
    region_sum = df.groupby(COL_REGION, as_index=False)[COL_CONF].sum().sort_values(COL_CONF, ascending=False)
    print("\nConfirmed by WHO Region:\n", region_sum)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=COL_REGION, y=COL_CONF, data=region_sum)
    plt.title("Confirmed Cases by WHO Region")
    plt.xticks(rotation=45, ha="right")
    save_show("who_region_confirmed")

# Scatter relationships 
plt.figure(figsize=(7, 5))
sns.scatterplot(x=COL_CONF, y=COL_REC, data=df)
plt.title("Confirmed vs Recovered")
save_show("scatter_confirmed_vs_recovered")

plt.figure(figsize=(7, 5))
sns.scatterplot(x=COL_CONF, y=COL_DEATHS, data=df)
plt.title("Confirmed vs Deaths")
save_show("scatter_confirmed_vs_deaths")