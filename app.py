from shiny import *
from shiny import reactive
from shared import df, batter_df, pitcher_df
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as patches

import pandas as pd

app_ui = ui.page_fluid(
    ui.panel_title("MLB Statcast Pitch Analysis", "Pitch Analysis"),
    # histogram
    # ui.input_select("var", "Select variable here", choices=list(df.columns)),
    # ui.panel_main(ui.output_plot("plot")),
    # pitch location by pitcher
    # Pitcher strike zone plot
    ui.card(
        ui.row(
            ui.column(
                6,
                ui.input_selectize(
                    "selected_pitcher",
                    "Select pitcher",
                    choices=sorted(df["pitcher_name"].unique().tolist()),
                    multiple=True,
                ),
                ui.input_selectize(
                    "selected_pitcher_opponents",
                    "Select batter(s) faced",
                    choices=[],
                    multiple=True,
                ),
                ui.input_selectize(
                    "selected_event_pitcher",
                    "Select pitch outcome (event)",
                    choices=sorted(df["events"].dropna().unique().tolist()),
                    multiple=True,
                ),
                ui.input_selectize(
                    "selected_description_pitcher",
                    "Select pitch outcome (description)",
                    choices=sorted(df["description"].dropna().unique().tolist()),
                    multiple=True,
                ),
            ),
            ui.column(6, ui.output_plot("strike_zone_plot_pitcher")),
            style="margin-left: 10%; margin-right: 10%; max-width: 80%;",
        )
    ),
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

    @reactive.Calc
    def batters_faced():
        if input.selected_pitcher():
            return sorted(
                df[df["pitcher_name"] == input.selected_pitcher()]["batter_name"]
                .unique()
                .tolist()
            )

    @reactive.Effect
    def update_pitcher_opponents():
        ui.update_selectize("selected_pitcher_opponents", choices=batters_faced())

    @reactive.Calc
    def pitcher_pitches_filtered():
        print(df.shape[0])
        filter_criteria = {
            "pitcher_name": input.selected_pitcher(),
            "batter_name": input.selected_pitcher_opponents(),
            "events": input.selected_event_pitcher(),
            "description": input.selected_description_pitcher(),
        }
        if all(not criteria for criteria in filter_criteria.values()):
            return None
        compound_condition = pd.Series([True] * len(df), index=df.index)
        for column, filter in filter_criteria.items():
            if filter:
                if isinstance(filter, str):
                    condition = df[column] == filter
                else:
                    condition = df[column].isin(filter)
                compound_condition &= condition
        filtered_pitches = df[compound_condition]
        return filtered_pitches

    @output
    @render.plot
    def strike_zone_plot_pitcher():
        fig, ax = create_strike_zone()
        data = pitcher_pitches_filtered()
        if data is None:
            ax.text(0, 4, "Please select a filter", ha='center', va='center')
            return
        if data.empty:
            ax.text(0, 4, "No such pitch on record", ha='center', va='center')
        else:
            labels, pitch_types = np.unique(data["pitch_type"], return_inverse=True)
            scatter = ax.scatter(
                data["plate_x"],
                data["plate_z"],
                c=pitch_types,
                s=3,
                marker="o",
                cmap="Dark2",
            )
            ax.set_title(f"Pitch Locations of {input.selected_pitcher()}")
            ax.legend(scatter.legend_elements()[0], labels)

    '''
    @output
    @render.plot
    def strike_zone_plot_batter():
        plt, ax = create_strike_zone()
        data = df[
            (df["batter_name"] == input.selected_batter())
            & (df["events"] == input.selected_outcome_batter())
        ][["plate_x", "plate_z"]].dropna()
        ax.scatter(data["plate_x"], data["plate_z"], s=1)
        ax.set_title(
            f"Pitch Location of {input.selected_batter()} That Resulted in {input.selected_outcome_batter()}"
        )
    '''


app = App(app_ui, server, debug=True)


"""
    ui.card(
        ui.row(
            ui.column(
                6,
                ui.input_selectize(
                    "selected_batter",
                    "Select batter",
                    choices=sorted(df["batter_name"].unique().tolist()),
                ),
                ui.input_selectize(
                    "selected_outcome_batter",
                    "Select pitch outcome",
                    choices=sorted(df["events"].dropna().unique().tolist()),
                    multiple=True,
                ),
            ),
            ui.column(6, ui.output_plot("strike_zone_plot_batter")),
            style="margin-left: 10%; margin-right: 10%; max-width: 80%;",
        ),
    ),
"""
