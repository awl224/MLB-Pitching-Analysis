from shiny import *
from shiny import reactive
from import_data import df
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sportypy.surfaces.baseball import MLBField

import matplotlib.patches as patches
import pandas as pd

plots_dir = Path(__file__).parent / "plots"


page1 = ui.page_fluid(
    ui.panel_title("Data & Analytics", "MLB Pitch Analysis"),
    # histogram
    # ui.input_select("var", "Select variable here", choices=list(df.columns)),
    # ui.panel_main(ui.output_plot("plot")),
    # pitch location by pitcher
    # Pitcher strike zone plot
    ui.card(
        ui.row(
            ui.column(
                4,
                ui.input_selectize(
                    "selected_pitcher",
                    "Select pitcher",
                    choices=[],
                    multiple=True,
                ),
                ui.input_selectize(
                    "selected_batter",
                    "Select batter(s) faced",
                    choices=[],
                    multiple=True,
                ),
                ui.input_selectize(
                    "selected_event",
                    "Select pitch outcome (event)",
                    choices=[],
                    multiple=True,
                ),
                ui.input_selectize(
                    "selected_description",
                    "Select pitch outcome (description)",
                    choices=[],
                    multiple=True,
                ),
            ),
            ui.column(4, ui.output_plot("strike_zone_plot_pitcher")),
            ui.column(4, ui.output_plot("hit_location_plot")),
        ),
    ),
    ui.card(
        ui.row(
            ui.column(6, ui.output_plot("pitch_type_distribution")),
            ui.column(
                6,
                ui.output_plot("pitch_speed_histogram"),
                ui.input_selectize(
                    "selected_pitch_types",
                    "Select pitch type",
                    choices=[],
                    multiple=True,
                ),
            ),
        )
    ),
)

page2 = ui.page_fillable(
    ui.panel_title("Blake Snell Pitch Type Prediction", "Pitch Prediction"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Introduction"),
            ui.p(
                "In this analysis, the models attempt to predict the pitch type thrown by MLB pitcher Blake Snell during the 2023 season. Snell threw four different pitch types in 2023 including fastballs (FF), curveballs (CU), sliders (SL), and changeups (CH). Accurately predicting pitch type is a valuable opportunity for game strategy and coaching, allowing teams to better anticipate specific situations and pitches."
            ),
            ui.p(
                "The dataset used includes all information leading up to the pitch, such as the ball-strike count, the inning, the number of outs, the current batter, and runners on base. However, only data available before the pitch is included in the training set, ensuring the model predicts based on the same information available to players and coaches in real-time. Excluding data from after the pitch, such as pitch outcome, speed, and vertical/horizontal break, make sure the models focus only on factors influencing the decision-making process before the pitch is delivered."
            ),
            ui.p(
                "This project highlights the challenges of multiclass classification, where the target variable (pitch type) has more than two possible outcomes. In Snell’s pitches, the dataset is also imbalanced, as he throws certain pitches more than others. This poses a unique challenge for models and makes it important to evaluate performance past just accuracy."
            ),
            ui.p(),
        ),
        ui.card(
            ui.card_header("Model Description"),
            ui.p(
                "Models used include a Decision Tree classifier and a K-nearest neighbors (KNN) classifier. Two dummy classifiers were included as a baseline: one with a stratified random method and the other with a zero-rule strategy, which consistently predicted the most frequent class."
            ),
            ui.p(
                "To ensure comprehensive performance measurements, the model evaluation process involved five repetitions of 10-fold cross-validation. Cross-validation is a crucial stage since it lowers the possibility of overfitting and can provide a more comprehensive assessment of the model’s abilities. We make sure that our findings are independent of a specific train-test split by repeating cross-validation, which proves a more accurate assessment of performance."
            ),
            ui.p(
                "Multiple performance metrics include accuracy, precision, recall, and F1-score. Although accuracy gives an overall idea of how frequently the model predicts the correct pitch type, imbalanced datasets like this one can make accuracy misleading. As a result, the analysis benefits from precision, recall, and F1-score, which demonstrate the models' ability to classify correctly."
            ),
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.output_image("precision_image", height="100%"),
            fill=True,
            full_screen=True,
        ),
        ui.card(
            ui.output_image("recall_image", height="100%"),
            fill=True,
            full_screen=True,
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.output_image("f1_image", height="100%"), fill=True, full_screen=True
        ),
        ui.card(
            ui.output_image("accuracy_image", height="100%"),
            fill=True,
            full_screen=True,
        ),
    ),
)

app_ui = ui.page_navbar(
    ui.nav_panel("Data & Analytics", page1),
    ui.nav_panel("Pitch Type Prediction ML", page2),
    title="2023 MLB Statcast Pitch Analytics",
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
            zorder=10,
        )
        ax.add_patch(strike_zone)
        return fig, ax

    @reactive.Calc
    def pitcher_reactive():
        filter_conditions = {
            "batter_name": input.selected_batter(),
            "events": input.selected_event(),
            "description": input.selected_description(),
        }
        filtered_df = df.copy()
        for column, selected_values in filter_conditions.items():
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        return sorted(filtered_df["pitcher_name"].dropna().unique().tolist())

    @reactive.Effect
    def update_pitcher():
        ui.update_selectize(
            "selected_pitcher",
            choices=pitcher_reactive(),
            selected=input.selected_pitcher(),
        )

    @reactive.Calc
    def batter_reactive():
        filter_conditions = {
            "pitcher_name": input.selected_pitcher(),
            "events": input.selected_event(),
            "description": input.selected_description(),
        }
        filtered_df = df.copy()
        for column, selected_values in filter_conditions.items():
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        return sorted(filtered_df["batter_name"].dropna().unique().tolist())

    @reactive.Effect
    def update_pitcher_opponents():
        ui.update_selectize(
            "selected_batter",
            choices=batter_reactive(),
            selected=input.selected_batter(),
        )

    @reactive.Calc
    def event_reactive():
        filter_conditions = {
            "pitcher_name": input.selected_pitcher(),
            "batter_name": input.selected_batter(),
            "description": input.selected_description(),
        }
        filtered_df = df.copy()
        for column, selected_values in filter_conditions.items():
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

        return sorted(filtered_df["events"].dropna().unique().tolist())

    @reactive.Effect
    def update_events():
        ui.update_selectize(
            "selected_event",
            choices=event_reactive(),
            selected=input.selected_event(),
        )

    @reactive.Calc
    def description_reactive():
        filter_conditions = {
            "pitcher_name": input.selected_pitcher(),
            "batter_name": input.selected_batter(),
            "events": input.selected_event(),
        }
        filtered_df = df.copy()
        for column, selected_values in filter_conditions.items():
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        return sorted(filtered_df["description"].dropna().unique().tolist())

    @reactive.Effect
    def update_description():
        ui.update_selectize(
            "selected_description",
            choices=description_reactive(),
            selected=input.selected_description(),
        )

    @reactive.Calc
    def pitcher_pitches_filtered():
        print(df.shape[0])
        filter_criteria = {
            "pitcher_name": input.selected_pitcher(),
            "batter_name": input.selected_batter(),
            "events": input.selected_event(),
            "description": input.selected_description(),
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
        ax.set_title(f"Pitch Locations")
        if data is None:
            ax.text(0, 4, "Please select a filter", ha="center", va="center")
            return
        if data.empty:
            ax.text(0, 4, "No data available", ha="center", va="center")
            return

        labels, pitch_types = np.unique(data["pitch_type"], return_inverse=True)
        scatter = ax.scatter(
            data["plate_x"],
            data["plate_z"],
            c=pitch_types,
            s=3,
            marker="o",
            cmap="Dark2",
            zorder=5,
        )
        ax.legend(scatter.legend_elements()[0], labels)
        num_pitches = len(data)
        ax.set_title(f"{num_pitches} Pitch Locations")

    @output
    @render.plot
    def hit_location_plot():
        # field = MLBField()
        field_parameters = {
            "field_units": "ft",
            "left_field_distance": 355.0,
            "right_field_distance": 355.0,
            "center_field_distance": 400.0,
            "baseline_distance": 90.0,
            "running_lane_start_distance": 45.0,
            "running_lane_depth": 3.0,
            "running_lane_length": 48.0,
            "pitchers_mound_center_to_home_plate": 46.0,
            "pitchers_mound_radius": 5.0,
            "pitchers_plate_front_to_home_plate": 47.0,
            "pitchers_plate_width": 0.5,
            "pitchers_plate_length": 2.0,
            "base_side_length": 1.25,
            "home_plate_edge_length": 1.4167,
            "infield_arc_radius": 95.0,
            "base_anchor_to_infield_grass_radius": 13.0,
            "line_width": 0.25,
            "foul_line_to_infield_grass": 3.0,
            "foul_line_to_foul_grass": 3.0,
            "batters_box_length": 6.0,
            "batters_box_width": 4.0,
            "batters_box_y_adj": 0.7083,
            "home_plate_side_to_batters_box": 0.5,
            "catchers_box_shape": "trapezoid",
            "catchers_box_depth": 8.0,
            "catchers_box_width": 3.5833,
            "backstop_radius": 60.0,
            "home_plate_circle_radius": 9.0,
        }

        ax = MLBField(field_updates=field_parameters).draw(
            xlim=(-500.0, 500.0), ylim=(-20.0, 500.0)
        )
        data = pitcher_pitches_filtered()

        if data is None:
            return
        if data.empty:
            return
        data = data.dropna(subset=["hc_x", "hc_y"])
        if data is None:
            return
        if data.empty:
            return
        labels, hit_types = np.unique(data["bb_type"], return_inverse=True)
        scatter = ax.scatter(
            2 * (data["hc_x"] - 126),
            2 * (207 - data["hc_y"]),
            c=hit_types,
            s=3,
            marker="o",
            cmap="Dark2",
            zorder=5,
        )
        ax.legend(scatter.legend_elements()[0], labels)
        num_pitches = len(data)
        ax.set_title(f"{num_pitches} Hit Locations")

    @output
    @render.plot
    def pitch_type_distribution():
        fig, ax = plt.subplots(figsize=(10, 5))
        data = pitcher_pitches_filtered()
        ax.set_title("Pitch Type")

        if data is None:
            ax.text(
                0.5,
                0.5,
                "Please select a filter",
                ha="center",
                va="center",
                fontsize=12,
            )
            return
        if data.empty:
            ax.text(
                0.5, 0.5, "No data available", ha="center", va="center", fontsize=12
            )
            return

        pitch_types = data["pitch_type"].dropna().value_counts()
        min_colormap_val = pitch_types.values.min()
        max_colormap_val = pitch_types.values.max()
        offset = 0.2 * max_colormap_val
        norm = Normalize(vmin=min_colormap_val - offset, vmax=max_colormap_val)
        cmap = plt.get_cmap("Blues")

        bars = ax.bar(
            pitch_types.index, pitch_types.values, color=cmap(norm(pitch_types.values))
        )

        total = pitch_types.values.sum()
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value}",
                ha="center",
                va="bottom",
            )

            percent = "%0.0f%%" % (100 * float(value) / total)
            ax.annotate(
                percent,
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xycoords="data",
                xytext=(0, -20),
                textcoords="offset points",
                va="top",
                ha="center",
                weight="bold",
                color="black",
            )

        plt.subplots_adjust(bottom=0.1)

    @output
    @render.plot
    def pitch_speed_histogram():
        fig, ax = plt.subplots(figsize=(10, 5))
        data = pitcher_pitches_filtered()
        pitch_type_selected = input.selected_pitch_types()
        if pitch_type_selected:
            data = data[data["pitch_type"].isin(pitch_type_selected)]
        ax.set_title("Pitch Speed (mph)")

        if data is None:
            ax.text(
                0.5,
                0.5,
                "Please select a filter",
                ha="center",
                va="center",
                fontsize=12,
            )
            return
        if data.empty:
            ax.text(
                0.5, 0.5, "No data available", ha="center", va="center", fontsize=12
            )
            return

        release_speeds = data["release_speed"].dropna()
        counts, bins, patches = ax.hist(
            release_speeds, bins=10, color="blue", edgecolor="black"
        )

        min_colormap_val = min(counts)
        max_colormap_val = max(counts)
        offset = 0.2 * max_colormap_val
        norm = Normalize(vmin=min_colormap_val, vmax=max_colormap_val)
        cmap = plt.get_cmap("YlOrRd")
        ax.set_xticks(bins)

        total = counts.sum()
        for count, patch in zip(counts, patches):
            plt.setp(patch, "facecolor", cmap(norm(count)))
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                str(int(count)),
                ha="center",
                va="bottom",
                color="black",
            )

            percent = "%0.0f%%" % (100 * float(count) / total)
            ax.annotate(
                percent,
                xy=(patch.get_x() + patch.get_width() / 2, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -20),
                textcoords="offset points",
                va="top",
                ha="center",
                weight="bold",
                color="black",
            )

        plt.subplots_adjust(bottom=0.1)

    @reactive.Effect
    def update_pitch_types():
        data = pitcher_pitches_filtered()
        choices_to_set = []
        if data is not None:
            choices_to_set = sorted(
                pitcher_pitches_filtered()["pitch_type"].dropna().unique().tolist()
            )
        ui.update_selectize(
            "selected_pitch_types",
            choices=choices_to_set,
            selected=input.selected_pitch_types(),
        )

    @render.image
    def precision_image():
        img = {"src": str(plots_dir / "precision_score.png"), "width": "100%"}
        return img

    @render.image
    def recall_image():
        img = {"src": str(plots_dir / "recall.png"), "width": "100%"}
        return img

    @render.image
    def f1_image():
        img = {"src": str(plots_dir / "f_score.png"), "width": "100%"}
        return img

    @render.image
    def accuracy_image():
        img = {"src": str(plots_dir / "accuracy.png"), "width": "100%"}
        return img


app = App(app_ui, server, debug=True)
