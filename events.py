#Extracts events from original CSVs into a set to find different types of events

import pandas as pd

INPUT_CSV = "ILIData2007.csv"
COLUMN_NAME = "event"

df = pd.read_csv(INPUT_CSV, encoding="latin1")

unique_events = set()

for value in df[COLUMN_NAME]:
    if pd.notna(value):          # skip NaNs
        unique_events.add(value.strip())

print(unique_events)
