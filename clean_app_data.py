import os
import pandas as pd

input_directory = "./model_data/pitch_data/"
output_directory = "./app/pitch_data/"

columns_to_keep = [
    "sz_bot",
    "sz_top",
    "player_name",
    "events",
    "description",
    "pitch_type",
    "release_speed",
    "bb_type",
    "hc_x",
    "hc_y",
    "plate_x",
    "plate_z",
    "batter"
]

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)
        df_cleaned = df[columns_to_keep]
        output_file_path = os.path.join(output_directory, filename)
        df_cleaned.to_csv(output_file_path, index=False)
        print(f"Processed and saved {filename} to {output_file_path}")
