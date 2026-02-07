import pandas as pd

#2022
INPUT_CSV = "ILIData2022.csv"        # your source CSV
OUTPUT_CSV = "2022anomalies.csv"  # output CSV
COLUMN_NAME = "Event Description"   # column to search
KEYWORDS = [
    "dent",
    "metal loss",
    "metal loss manufacturing anomaly",
    "seam weld anomaly - b",
    "seam weld dent",
    "seam weld manufacturing anomaly",
    "cluster",
    "girth weld anomaly",
    "metal loss manufacturing",
    "metal loss-manufacturing anomaly"
]

# Read CSV
df = pd.read_csv(INPUT_CSV, encoding="latin1")
# Keep rows where COLUMN_NAME contains "metal loss" (case-insensitive)
pattern = "|".join(KEYWORDS)
filtered = df[
    df[COLUMN_NAME]
    .astype(str)
    .str.lower()
    .str.contains(pattern, na=False)
]
filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(filtered)} rows to {OUTPUT_CSV}")


#2015
INPUT_CSV = "ILIData2015.csv"        # your source CSV
OUTPUT_CSV = "2015anomalies.csv"  # output CSV
COLUMN_NAME = "Event Description"   # column to search
KEYWORDS = [
    "dent",
    "metal loss",
    "metal loss manufacturing anomaly",
    "seam weld anomaly - b",
    "seam weld dent",
    "seam weld manufacturing anomaly",
    "cluster",
    "girth weld anomaly",
    "metal loss manufacturing",
    "metal loss-manufacturing anomaly"
]

# Read CSV
df = pd.read_csv(INPUT_CSV, encoding="latin1")
# Keep rows where COLUMN_NAME contains "metal loss" (case-insensitive)
pattern = "|".join(KEYWORDS)
filtered = df[
    df[COLUMN_NAME]
    .astype(str)
    .str.lower()
    .str.contains(pattern, na=False)
]
filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(filtered)} rows to {OUTPUT_CSV}")


#2007
INPUT_CSV = "ILIData2007.csv"        # your source CSV
OUTPUT_CSV = "2007anomalies.csv"  # output CSV
COLUMN_NAME = "event"   # column to search
KEYWORDS = [
    "dent",
    "metal loss",
    "metal loss manufacturing anomaly",
    "seam weld anomaly - b",
    "seam weld dent",
    "seam weld manufacturing anomaly",
    "cluster",
    "girth weld anomaly",
    "metal loss manufacturing",
    "metal loss-manufacturing anomaly"
]

# Read CSV
df = pd.read_csv(INPUT_CSV, encoding="latin1")
# Keep rows where COLUMN_NAME contains "metal loss" (case-insensitive)
pattern = "|".join(KEYWORDS)
filtered = df[
    df[COLUMN_NAME]
    .astype(str)
    .str.lower()
    .str.contains(pattern, na=False)
]
filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(filtered)} rows to {OUTPUT_CSV}")

