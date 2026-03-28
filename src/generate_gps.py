# generate_gps.py

import os
import pandas as pd
import random

IMAGE_FOLDER = r"D:\LaneDetection\dataset\laneImagesTest"

# Base coordinates for multiple cities
city_locations = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867)
}

image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")],
    key=lambda x: int(x.replace("image","").replace(".jpg",""))
)

rows = []
cities = list(city_locations.keys())

for i, img in enumerate(image_files):

    # Rotate cities across images
    city = cities[i % len(cities)]
    base_lat, base_lon = city_locations[city]

    # Add small random variation for realism
    latitude = round(base_lat + random.uniform(-0.01, 0.01), 6)
    longitude = round(base_lon + random.uniform(-0.01, 0.01), 6)

    rows.append({
        "image_name": img,
        "city": city,
        "latitude": latitude,
        "longitude": longitude
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(IMAGE_FOLDER, "gps_data.csv"), index=False)

print("Multi-city GPS file created successfully.")