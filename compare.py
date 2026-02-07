import pandas as pd

# Load CSVs
df_2015 = pd.read_csv("2015anomalies.csv", encoding="latin1")
df_2022 = pd.read_csv("2022anomalies.csv", encoding="latin1")

# Exact column names (must match CSV headers)
CLOCK_2015 = 14
CLOCK_2022 = 18
EVENT_COL = "Event Description"
DISTUS_2015 = 4
DISTUS_2022 = 3
ODOMETER_2015 = 5
ODOMETER_2022 = 5

def time_to_decimal(s):
    try:
        h, m = str(s).strip().split(":")
        return int(h) + int(m) / 60.0
    except:
        return None

# Loop through anomalies
for i_2015, row_2015 in df_2015.iterrows():
    event_2015 = row_2015[EVENT_COL]
    clock_2015 = row_2015[CLOCK_2015]
    distUS_2015 = row_2015[DISTUS_2015]
    distUS_2015 = abs(float(distUS_2015))
    totalDist_2015 = row_2015[ODOMETER_2015]
    totalDist_2015 = float(totalDist_2015)
    jointNum_2015 = row_2015[0]

    if pd.isna(event_2015) or pd.isna(clock_2015):
        continue

    for i_2022, row_2022 in df_2022.iterrows():
        event_2022 = row_2022[EVENT_COL]
        clock_2022 = row_2022[CLOCK_2022]
        distUS_2022 = row_2022[DISTUS_2022]
        distUS_2022 = abs(float(distUS_2022))
        totalDist_2022 = row_2022[ODOMETER_2022]
        totalDist_2022 = float(totalDist_2022)
        jointNum_2022 = row_2022[0]

        if pd.isna(event_2022) or pd.isna(clock_2022):
            continue

        event_2015 = event_2015.lower()
        event_2022 = event_2022.lower()

        #clock_diff = abs(clock_2022 - clock_2015)
        if clock_2015 and clock_2022:
            clock_diff = abs(time_to_decimal(clock_2022) - time_to_decimal(clock_2015))
        else:
            clock_diff = 1000  # Large number to indicate missing clock

        try:
            jointNum_diff = abs(jointNum_2022 - jointNum_2015)
        except:
            jointNum_diff = 1000

        distUS_diff = abs(distUS_2022 - distUS_2015)
        totalDist_diff = abs(totalDist_2022 - totalDist_2015)

        clock_tolerance = 0.1
        totalDist_tolerance = 40
        distUS_tolerance = 5
        jointNum_tolerance = 50
                
        if event_2015 == event_2022 and clock_diff < clock_tolerance and totalDist_diff < totalDist_tolerance and distUS_diff < distUS_tolerance and jointNum_diff < jointNum_tolerance:
            print(
                f"MATCH: 2015[{i_2015+2}] and 2022[{i_2022+2}] → "
                f"Event='{event_2015}', Clock2015={clock_2015}, Clock2022={clock_2022}"
            )
