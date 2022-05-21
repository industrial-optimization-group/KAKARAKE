from typing import List, Union

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from matplotlib import cm
from scipy.stats import pearsonr
from tsp_solver.greedy import solve_tsp

from kakarake.clustering import cluster


def SCORE_bands(
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
        if len(groups) <= 8:
            colorscale = cm.get_cmap("Accent", len(groups))
            # print("hi!")
        else:
            colorscale = cm.get_cmap("tab20", len(groups))
    # colorscale = cm.get_cmap("viridis_r", len(groups))
    data = data * axis_signs
    num_labels = 6
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
        num_solns = len(solns["group"].values)
        r, g, b, a = colorscale(cluster_id)
        a = 0.6
        a_soln = 0.6
        color = f"rgba({r}, {g}, {b}, {a})"
        color_soln = f"rgba({r}, {g}, {b}, {a_soln})"
        low = solns.drop("group", axis=1).quantile(0.25)  # TODO Change back to 25/75
        high = solns.drop("group", axis=1).quantile(0.75)
        median = solns.drop("group", axis=1).median()
        if bands is True:
            # lower bound of the band
            fig.add_scatter(
                x=axis_positions,
                y=low,
                line={"color": color},
                name=f"50% band: Cluster {cluster_id}; {num_solns} Solutions        ",
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
                fillcolor=color,
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
                    line={"color": color_soln},
                    name=f"Solutions: Cluster {cluster_id}              ",
                    legendgroup=f"Solutions: Cluster {cluster_id}",
                    showlegend=legend,
                    visible=show_solutions,
                )
                legend = False
    # Axis lines
    for i, col_name in enumerate(column_names):
        # better = "Upper" if axis_signs[i] == -1 else "Lower"
        label_text = np.linspace(
            scales[col_name]["min"], scales[col_name]["max"], num_labels
        )
        # label_text = ["{:.3g}".format(i) for i in label_text]
        heights = np.linspace(0, 1, num_labels)
        scale_factors = []
        for current_label in label_text:
            try:
                with np.errstate(divide="ignore"):
                    scale_factors.append(int(np.floor(np.log10(np.abs(current_label)))))
            except OverflowError:
                pass

        scale_factor = int(np.median(scale_factors))
        if scale_factor == -1:
            scale_factor = 0

        label_text = label_text / 10 ** (scale_factor)
        label_text = ["{:.1f}".format(i) for i in label_text]
        if scale_factor != 0:
            scale_factor_text = f"e{scale_factor}"
        else:
            scale_factor_text = ""

        # Bottom axis label
        fig.add_scatter(
            x=[axis_positions[i]],
            y=[heights[0]],
            text=[label_text[0] + scale_factor_text],
            textposition="bottom center",
            mode="text",
            line={"color": "black"},
            showlegend=False,
        )
        # Top axis label
        fig.add_scatter(
            x=[axis_positions[i]],
            y=[heights[-1]],
            text=[label_text[-1] + scale_factor_text],
            textposition="top center",
            mode="text",
            line={"color": "black"},
            showlegend=False,
        )
        label_text[0] = ""
        label_text[-1] = ""
        # Intermediate axes labels
        fig.add_scatter(
            x=[axis_positions[i]] * num_labels,
            y=heights,
            text=label_text,
            textposition="middle left",
            mode="markers+lines+text",
            line={"color": "black"},
            showlegend=False,
        )

        fig.add_scatter(
            x=[axis_positions[i]],
            y=[1.10],
            text=f"{col_name}",
            textfont=dict(size=28),
            mode="text",
            showlegend=False,
        )
        """fig.add_scatter(
            x=[axis_positions[i]], y=[1.1], text=better, mode="text", showlegend=False,
        )
        fig.add_scatter(
            x=[axis_positions[i]],
            y=[1.05],
            text="is better",
            mode="text",
            showlegend=False,
        )"""
    fig.update_layout(font_size=18)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", font=dict(size=24)))
    return fig


def annotated_heatmap(correlation_matrix, col_names, order):
    corr = pd.DataFrame(correlation_matrix, index=col_names, columns=col_names)
    corr = corr[col_names[order]].loc[col_names[order[::-1]]]
    # corr = np.abs(np.rint(corr * 100) / 100)  # TODO UNDO
    corr = np.rint(corr * 100) / 100
    fig = ff.create_annotated_heatmap(
        corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.astype(str).values,
    )
    fig.update_layout(title="True correlations")
    return fig


def order_objectives(data: pd.DataFrame, use_absolute_corr: bool = False):
    # Calculating correlations
    # corr = spearmanr(data).correlation  # Pearson's coeff is better than Spearmann's, in some cases
    corr = np.asarray(
        [
            [
                pearsonr(data.values[:, i], data.values[:, j])[0]
                for j in range(len(data.columns))
            ]
            for i in range(len(data.columns))
        ]
    )
    # axes order: solving TSP
    distances = corr
    if use_absolute_corr:
        distances = np.abs(distances)
    obj_order = solve_tsp(-distances)
    return corr, obj_order


def calculate_axes_positions(
    data, obj_order, corr, dist_parameter, distance_formula: int = 1
):
    # axes positions
    order = np.asarray(list((zip(obj_order[:-1], obj_order[1:]))))
    axis_len = corr[order[:, 0], order[:, 1]]
    if distance_formula == 1:
        axis_len = 1 - axis_len  # TODO Make this formula available to the user
    elif distance_formula == 2:
        axis_len = 1 / np.abs(axis_len)  #  Reciprocal for reverse
    else:
        raise ValueError("distance_formula should be either 1 or 2 (int)")
    # axis_len = np.abs(axis_len)
    # axis_len = axis_len / sum(axis_len) #TODO Changed
    axis_len = axis_len + dist_parameter  # Minimum distance between axes
    axis_len = axis_len / sum(axis_len)
    axis_dist = np.cumsum(np.append(0, axis_len))
    # Axis signs (normalizing negative correlations)
    axis_signs = np.cumprod(np.sign(np.hstack((1, corr[order[:, 0], order[:, 1]]))))
    return data.iloc[:, obj_order], axis_dist, axis_signs


def auto_SCORE(
    data: pd.DataFrame,
    solutions: bool = True,
    bands: bool = True,
    medians: bool = False,
    dist_parameter: float = 0.05,
    use_absolute_corr: bool = False,
    distance_formula: int = 1,
    flip_axes: bool = False,
    clustering_algorithm: str = "DBSCAN",
    clustering_score: str = "silhoutte",
):
    # Calculating correlations and axes positions
    corr, obj_order = order_objectives(data, use_absolute_corr=use_absolute_corr)

    ordered_data, axis_dist, axis_signs = calculate_axes_positions(
        data,
        obj_order,
        corr,
        dist_parameter=dist_parameter,
        distance_formula=distance_formula,
    )
    if not flip_axes:
        axis_signs = None
    groups = cluster(
        ordered_data, algorithm=clustering_algorithm, score=clustering_score
    )
    groups = groups - np.min(groups)  # translate minimum to 0.
    fig1 = SCORE_bands(
        ordered_data,
        color_groups=groups,
        axis_positions=axis_dist,
        axis_signs=axis_signs,
        solutions=solutions,
        bands=bands,
        medians=medians,
    )
    return fig1, corr, obj_order, groups
