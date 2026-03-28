# ==========================================
# geo_sender.py
# Logs GPS Alerts
# ==========================================

import pandas as pd
import os
import datetime


def send_to_authority(latitude, longitude, visibility):

    timestamp = datetime.datetime.now()

    print("\nALERT SENT TO THE AUTHORITY")
    print("Location:", latitude, longitude)
    print("Visibility:", round(visibility, 3))
    print("Time:", timestamp)

    log_file = "alerts_log.csv"

    new_row = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "visibility": round(float(visibility), 3),
        "timestamp": timestamp
    }])

    if os.path.exists(log_file):
        new_row.to_csv(log_file, mode='a', header=False, index=False)
    else:
        new_row.to_csv(log_file, index=False)