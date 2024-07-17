from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


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
df = df.dropna(subset=["pitch_type","plate_x","plate_z"])
event_replacements = {
    "catcher_interf": "catcher interference",
    "caught_stealing_2b": "caught stealing 2b",
    "caught_stealing_3b": "caught stealing 3b",
    "caught_stealing_home": "caught stealing home",
    "double_play": "double play",
    "field_error": "field error",
    "field_out": "field out",
    "fielders_choice": "fielders choice",
    "fielders_choice_out": "fielders choice out",
    "force_out": "force out",
    "grounded_into_double_play": "grounded into double play",
    "hit_by_pitch": "hit by pitch",
    "home_run": "home run",
    "other_out": "other out",
    "pickoff_1b": "pickoff 1b",
    "pickoff_2b": "pickoff 2b",
    "pickoff_3b": "pickoff 3b",
    "pickoff_caught_stealing_2b": "pickoff caught stealing 2b",
    "sac_bunt": "sac bunt",
    "sac_fly": "sac fly",
    "sac_fly_double_play": "sac fly double play",
    "strikeout_double_play": "strikeout double play",
    "wild_pitch": "wild pitch",
}
df["events"] = df["events"].replace(event_replacements)

description_replacements = {
    "blocked_ball": "blocked ball",
    "bunt_foul_tip": "bunt foul tip",
    "called_strike": "called strike",
    "foul_bunt": "foul bunt",
    "foul_tip": "foul tip",
    "hit_by_pitch": "hit by pitch",
    "hit_into_play": "hit into play",
    "missed_bunt": "missed bunt",
    "swinging_strike": "swinging strike",
    "swinging_strike_blocked": "swinging strike blocked",
}
df["description"] = df["description"].replace(description_replacements)

# Check that all pitches had a valid batter
# print(df.shape[0] == pitcher_df.shape[0])

# print(df["sz_top"].mean())
