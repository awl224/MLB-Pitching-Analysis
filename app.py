from shiny import *
from shiny import reactive

from shared import df, batter_df, pitcher_df
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import pandas

app_ui = ui.page_fluid(
    ui.card(
        ui.input_select("var", "Select variable here", choices=list(df.columns)),
        ui.panel_main(ui.output_plot("plot")),
        ui.input_selectize(
            "selected_pitcher",
            "Select pitcher here",
            choices=sorted(df["pitcher_name"].unique().tolist()),
        ),
        ui.panel_main(ui.output_plot("strike_zone_plot")),
        ui.input_selectize(
            "selected_batter",
            "Select batter here",
            choices=sorted(df["batter_name"].unique().tolist()),
        ),
        ui.input_selectize(
            "selected_outcome",
            "Select pitch outcome here",
            choices=sorted(df["events"].dropna().unique().tolist()),
        ),
        ui.panel_main(ui.output_plot("strike_zone_plot_batter")),
        str(df["plate_x"].describe()),
        str(df["plate_z"].describe()),
        df.head(),
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.plot
    def plot():
        var_selected = input.var()
        data = df[var_selected].dropna()
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color="blue", edgecolor="black", alpha=0.7)
        plt.xlabel(var_selected)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {var_selected}")

    @output
    @render.plot
    def strike_zone_plot():
        plt.figure(figsize=(5, 5))
        data = df[df["pitcher_name"] == input.selected_pitcher()][
            ["plate_x", "plate_z"]
        ].dropna()
        plt.scatter(data["plate_x"], data["plate_z"], s=1)
        plt.xlabel("Plate x")
        plt.ylabel("Plate y")
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3, 8)
        plt.grid(True)
        plt.title("Pitch Location")

    @output
    @render.plot
    def strike_zone_plot_batter():
        plt.figure(figsize=(5, 5))
        data = df[
            (df["batter_name"] == input.selected_batter())
            & (df["events"] == input.selected_outcome())
        ][["plate_x", "plate_z"]].dropna()
        plt.scatter(data["plate_x"], data["plate_z"], s=1)
        plt.xlabel("Plate x")
        plt.ylabel("Plate y")
        plt.xlim(-3.5, 3.5)
        plt.ylim(-2, 6)
        plt.grid(True)
        plt.title("Pitch Location")


app = App(app_ui, server, debug=True)
