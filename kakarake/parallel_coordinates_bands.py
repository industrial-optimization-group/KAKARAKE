from kakarake.parallel_coords import (
    parallel_coordinates,
    group_objectives,
    GaussianMixtureclustering,
)
import pandas as pd
import numpy as np
import plotly.express as ex
import plotly.graph_objects as go
from typing import Union, List
from matplotlib import cm
from scipy.stats import spearmanr
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    OPTICS,
    Birch,
    MeanShift,
    AffinityPropagation,
    AgglomerativeClustering,
    SpectralClustering,
)

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


def parallel_coordinates_bands_lines(
    data: pd.DataFrame,
    axis_signs=None,
    color_groups: Union[List, bool] = None,
    axis_positions: np.ndarray = None,
):
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
    scaled_data = data - data.min()
    scaled_data = scaled_data / scaled_data.max()
    scales = pd.DataFrame([data.min(), data.max()], index=["min", "max"]) * axis_signs

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
        # lower bound of the band
        fig.add_scatter(
            x=axis_positions,
            y=low,
            line={"color": color},
            name=f"Cluster {cluster_id}: 50% band",
            mode="lines",
            legendgroup=f"Cluster {cluster_id}: 50% band",
            showlegend=True,
            line_shape="spline",
        )
        # upper bound of the band
        fig.add_scatter(
            x=axis_positions,
            y=high,
            line={"color": color},
            name="50% band upper",
            mode="lines",
            legendgroup=f"Cluster {cluster_id}: 50% band",
            showlegend=False,
            line_shape="spline",
            fill="tonexty",
        )
        # median
        fig.add_scatter(
            x=axis_positions,
            y=median,
            line={"color": color},
            name=f"Cluster {cluster_id}: Median",
            mode="lines+markers",
            marker=dict(line=dict(color="Black", width=2)),
            legendgroup=f"Cluster {cluster_id}: Median",
            showlegend=True,
        )
        # individual solutions
        legend = True
        for _, soln in solns.drop("group", axis=1).iterrows():
            fig.add_scatter(
                x=axis_positions,
                y=soln,
                line={"color": color},
                name=f"Cluster {cluster_id}: solutions",
                legendgroup=f"Cluster {cluster_id}: solutions",
                showlegend=legend,
            )
            legend = False
    # Axis lines
    for i, col_name in enumerate(column_names):
        better = "Lower" if axis_signs[i] == -1 else "Upper"
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
            text=f"<b>Objective {col_name}</b>",
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


def auto_par_coords(data: pd.DataFrame):
    # Color grouping
    groups = GaussianMixtureclustering(data)
    obj_order = group_objectives(data)
    col_order = [data.columns[i] for i in obj_order]
    # Calculating correlations
    data = data[col_order]
    corr = spearmanr(data).correlation
    # axis positions
    axis_len = np.asarray([corr[i, i + 1] for i in range(len(data.columns) - 1)])
    axis_len = 1 / np.abs(axis_len)  #  Reciprocal for reverse
    axis_len = axis_len / sum(axis_len)
    axis_len = axis_len + 0.15  # Minimum distance between axes
    axis_len = axis_len / sum(axis_len)
    axis_dist = np.cumsum(np.append(0, axis_len))
    # Axis signs (normalizing negative correlations)
    axis_signs = np.cumprod(
        np.sign(
            np.hstack(
                (1, np.asarray([corr[i, i + 1] for i in range(len(data.columns) - 1)]))
            )
        )
    )
    return parallel_coordinates_bands_lines(
        data, color_groups=groups, axis_positions=axis_dist, axis_signs=axis_signs
    )
