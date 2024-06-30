from pathlib import Path

import pandas as pd

# Set up path and load CSV files containing pitch data
data_dir = Path(__file__).parent / "pitch_data"
csv_files = list(data_dir.glob("*.csv"))

# List of data frames
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all data frames in list
pitcher_df = pd.concat(dfs, ignore_index=True)
pitcher_df.rename(columns={"player_name": "pitcher_name"}, inplace=True)

# Dropping deprecated/unnecessary columns
pitcher_df = pitcher_df.drop(
    columns=[
        "spin_dir",
        "spin_rate_deprecated",
        "break_angle_deprecated",
        "break_length_deprecated",
        "tfs_deprecated",
        "tfs_zulu_deprecated",
        "umpire",
        "sv_id",
        "bat_speed",
        "swing_length",
    ]
)

# Load batter data
batter_dir = Path(__file__).parent / "batter_data"
batter_df = pd.read_csv(batter_dir/"batters_2023.csv")

# Remove all columns except for id and name
batter_df = batter_df[["player_id","player_name"]]
batter_df.rename(columns={'player_name':'batter_name','player_id':'batter'}, inplace=True)

# Merge pitcher and batter data frames (now have batter name for all pitches)
df = pd.merge(pitcher_df, batter_df, on="batter")

# Check that all pitches had a valid batter
#print(df.shape[0] == pitcher_df.shape[0])

#print(df["sz_top"].mean())
