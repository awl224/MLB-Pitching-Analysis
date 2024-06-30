from shiny import *
from shiny import reactive
from shared import df, batter_df, pitcher_df
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as patches

import pandas

app_ui = ui.page_fluid(
    ui.card(
        # histogram
        # ui.input_select("var", "Select variable here", choices=list(df.columns)),
        # ui.panel_main(ui.output_plot("plot")),
        # pitch location by pitcher
        ui.input_selectize(
            "selected_pitcher",
            "Select pitcher here",
            choices=sorted(df["pitcher_name"].unique().tolist()),
        ),
        ui.panel_main(ui.output_plot("strike_zone_plot_pitcher")),
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
        # str(df["plate_x"].describe()),
        # str(df["plate_z"].describe()),
        # df.head(),
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    # def setup_strike_zone_plot(ax, zone_bottom, zone_top):

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

    def create_strike_zone():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect("equal")
        ax.set(
            xlim=(-3, 3),
            ylim=(0, 5),
            xlabel=("Plate X"),
            ylabel=("Plate Y"),
            autoscale_on=False,
        )
        # Overlay a red rectangle for the strike zone
        strike_zone = patches.Rectangle(
            (-0.708335, df["sz_bot"].mean()),
            1.41667,
            (df["sz_top"].mean() - df["sz_bot"].mean()),
            fill=False,
            color="red",
            linewidth=1,
        )
        ax.add_patch(strike_zone)
        return fig, ax

    @output
    @render.plot
    def strike_zone_plot_pitcher():
        fig, ax = create_strike_zone()
        data = df[df["pitcher_name"] == input.selected_pitcher()][
            ["plate_x", "plate_z"]
        ].dropna()
        ax.scatter(data["plate_x"], data["plate_z"], s=1)  # Smaller dot size
        ax.set_title(f"Pitch Locations of {input.selected_pitcher()}")

    @output
    @render.plot
    def strike_zone_plot_batter():
        plt, ax = create_strike_zone()
        data = df[
            (df["batter_name"] == input.selected_batter())
            & (df["events"] == input.selected_outcome())
        ][["plate_x", "plate_z"]].dropna()
        ax.scatter(data["plate_x"], data["plate_z"], s=1)
        ax.set_title(
            f"Pitch Location of {input.selected_batter()} That Resulted in {input.selected_outcome()}"
        )


app = App(app_ui, server, debug=True)
