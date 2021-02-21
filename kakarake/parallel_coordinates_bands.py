from kakarake.parallel_coords import (
    parallel_coordinates,
    group_objectives,
    GaussianMixtureclustering,
)

import pandas as pd
import numpy as np
import plotly.express as ex
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Union, List
from matplotlib import cm
from scipy.stats import spearmanr

from tsp_solver.greedy import solve_tsp

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

import os


def parallel_coordinates_bands_lines(
    data: pd.DataFrame,
    axis_signs=None,
    color_groups: Union[List, bool] = None,
    axis_positions: np.ndarray = None,
    solutions: bool = True,
    bands: bool = False,
    medians: bool = False,
):
    # show on render
    show_solutions = "legendonly"
    if bands:
        show_medians = "legendonly"
    if medians:
        show_medians = True
    # pio.templates.default = "simple_white"
    column_names = data.columns
    num_columns = len(column_names)
    if axis_positions is None:
        axis_positions = np.linspace(0, 1, num_columns)
    if axis_signs is None:
        axis_signs = np.ones_like(axis_positions)
    if color_groups is None:
        color_groups = "continuous"
        colorscale = cm.get_cmap("viridis")
    elif isinstance(color_groups, (np.ndarray, list)):
        groups = list(np.unique(color_groups))
        groupsdict = dict(zip(groups, range(len(groups))))
        colorscale = cm.get_cmap("Accent", len(groups))
    data = data * axis_signs
    num_labels = 10
    # Scaling
    scaled_data = data - data.min(axis=0)
    scaled_data = scaled_data / scaled_data.max(axis=0)
    scales = (
        pd.DataFrame([data.min(axis=0), data.max(axis=0)], index=["min", "max"])
        * axis_signs
    )

    fig = go.Figure()
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    scaled_data.insert(0, "group", value=color_groups)
    for name, solns in scaled_data.groupby("group"):
        cluster_id = solns["group"].values[0]
        color = "rgba" + str(colorscale(cluster_id))
        low = solns.drop("group", axis=1).quantile(0.25)
        high = solns.drop("group", axis=1).quantile(0.75)
        median = solns.drop("group", axis=1).median()
        if bands is True:
            # lower bound of the band
            fig.add_scatter(
                x=axis_positions,
                y=low,
                line={"color": color},
                name=f"50% band: Cluster {cluster_id}",
                mode="lines",
                legendgroup=f"50% band: Cluster {cluster_id}",
                showlegend=True,
                line_shape="spline",
                hovertext=f"Cluster {cluster_id}",
            )
            # upper bound of the band
            fig.add_scatter(
                x=axis_positions,
                y=high,
                line={"color": color},
                name=f"Cluster {cluster_id}",
                mode="lines",
                legendgroup=f"50% band: Cluster {cluster_id}",
                showlegend=False,
                line_shape="spline",
                fill="tonexty",
                hovertext=f"Cluster {cluster_id}",
            )
        if medians is True:
            # median
            fig.add_scatter(
                x=axis_positions,
                y=median,
                line={"color": color},
                name=f"Median: Cluster {cluster_id}",
                mode="lines+markers",
                marker=dict(line=dict(color="Black", width=2)),
                legendgroup=f"Median: Cluster {cluster_id}",
                showlegend=True,
                visible=show_medians,
            )
        if solutions is True:
            # individual solutions
            legend = True
            for _, soln in solns.drop("group", axis=1).iterrows():
                fig.add_scatter(
                    x=axis_positions,
                    y=soln,
                    line={"color": color},
                    name=f"Solutions: Cluster {cluster_id}",
                    legendgroup=f"Solutions: Cluster {cluster_id}",
                    showlegend=legend,
                    visible=show_solutions,
                )
                legend = False
    # Axis lines
    for i, col_name in enumerate(column_names):
        better = "Upper" if axis_signs[i] == -1 else "Lower"
        label_text = np.linspace(
            scales[col_name]["min"], scales[col_name]["max"], num_labels
        )
        label_text = ["{:g}".format(float("{:.4g}".format(i))) for i in label_text]
        fig.add_scatter(
            x=[axis_positions[i]] * num_labels,
            y=np.linspace(0, 1, num_labels),
            text=label_text,
            textposition="middle left",
            mode="markers+lines+text",
            line={"color": "black"},
            showlegend=False,
        )
        fig.add_scatter(
            x=[axis_positions[i]],
            y=[1.15],
            text=f"<b>{col_name}</b>",
            mode="text",
            showlegend=False,
        )
        fig.add_scatter(
            x=[axis_positions[i]], y=[1.1], text=better, mode="text", showlegend=False,
        )
        fig.add_scatter(
            x=[axis_positions[i]],
            y=[1.05],
            text="is better",
            mode="text",
            showlegend=False,
        )
    return fig


def annotated_heatmap(correlation_matrix, col_names, order):
    corr = pd.DataFrame(correlation_matrix, index=col_names, columns=col_names)
    corr = corr[col_names[order]].loc[col_names[order[::-1]]]
    corr = np.abs(np.rint(corr * 100) / 100)
    return ff.create_annotated_heatmap(
        corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.astype(str).values,
    )


def auto_par_coords(
    data: pd.DataFrame,
    solutions: bool = True,
    bands: bool = False,
    medians: bool = False,
    dist_parameter: float = 0.05,
):
    # Calculating correlations
    corr = spearmanr(data).correlation
    original_order = data.columns
    # axes order: solving TSP
    distances = -np.abs(corr)
    obj_order = solve_tsp(distances)
    # axes positions
    order = np.asarray(list((zip(obj_order[:-1], obj_order[1:]))))
    axis_len = corr[order[:, 0], order[:, 1]]
    axis_len = 1 / np.abs(axis_len)  #  Reciprocal for reverse
    axis_len = axis_len / sum(axis_len)
    axis_len = axis_len + dist_parameter  # Minimum distance between axes
    axis_len = axis_len / sum(axis_len)
    axis_dist = np.cumsum(np.append(0, axis_len))
    # Axis signs (normalizing negative correlations)
    axis_signs = np.cumprod(np.sign(np.hstack((1, corr[order[:, 0], order[:, 1]]))))
    data = data.iloc[:, obj_order]
    groups = GaussianMixtureclustering(data)

    fig1 = parallel_coordinates_bands_lines(
        data,
        color_groups=groups,
        axis_positions=axis_dist,
        axis_signs=axis_signs,
        solutions=solutions,
        bands=bands,
        medians=medians,
    )
    # No axis inversion
    """fig2 = parallel_coordinates_bands_lines_v2(
        data,
        color_groups=groups,
        axis_positions=axis_dist,
        axis_signs=None,
        solutions=solutions,
        bands=bands,
        medians=medians,
    )"""
    # correlation matrix
    corr_fig = annotated_heatmap(corr, original_order, obj_order)
    return fig1, corr_fig


def order_objectives(data: pd.DataFrame):
    # Calculating correlations
    corr = spearmanr(data).correlation
    # axes order: solving TSP
    distances = -np.abs(corr)
    obj_order = solve_tsp(distances)
    return corr, obj_order


def calculate_axes_positions(data, obj_order, corr, dist_parameter):
    # axes positions
    order = np.asarray(list((zip(obj_order[:-1], obj_order[1:]))))
    axis_len = corr[order[:, 0], order[:, 1]]
    axis_len = 1 / np.abs(axis_len)  #  Reciprocal for reverse
    axis_len = axis_len / sum(axis_len)
    axis_len = axis_len + dist_parameter  # Minimum distance between axes
    axis_len = axis_len / sum(axis_len)
    axis_dist = np.cumsum(np.append(0, axis_len))
    # Axis signs (normalizing negative correlations)
    axis_signs = np.cumprod(np.sign(np.hstack((1, corr[order[:, 0], order[:, 1]]))))
    return data.iloc[:, obj_order], axis_dist, axis_signs


if __name__ == "__main__":
    # data = pd.read_csv("c432-110.csv", header=None, sep="\s+")[range(2, 11)]
    for filename in os.listdir("data"):
        data = pd.read_csv("data/" + filename)
        auto_par_coords(data, solutions=True, bands=True)[0].write_html(
            file="Outputs/" + filename.split(".")[0] + "_plot.html"
        )

