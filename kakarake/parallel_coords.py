import pandas as pd
import numpy as np
import plotly.express as ex
import plotly.graph_objects as go
import plotly.io as pio
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


def parallel_coordinates(
    data: pd.DataFrame,
    color_groups: Union[List, bool] = None,
    axis_positions: np.ndarray = None,
):
    # pio.templates.default = "simple_white"
    column_names = data.columns
    num_columns = len(column_names)
    if axis_positions is None:
        axis_positions = np.linspace(0, 1, num_columns)

    if color_groups is None:
        color_groups = "continuous"
        colorscale = cm.get_cmap("viridis")
    elif isinstance(color_groups, (np.ndarray, list)):
        groups = list(np.unique(color_groups))
        groupsdict = dict(zip(groups, range(len(groups))))
        colorscale = cm.get_cmap("Accent", len(groups))

    num_labels = 10
    # Scaling
    scaled_data = data - data.min()
    scaled_data = scaled_data / scaled_data.max()
    scales = pd.DataFrame([data.min(), data.max()], index=["min", "max"])

    fig = go.Figure()
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    for i, y in enumerate(scaled_data.values):
        show_legend = False
        if isinstance(color_groups, str):
            # Continuous
            color = "rgba" + str(colorscale(y[0]))
            group = None
        elif isinstance(color_groups, bool):
            # No color
            color = "gray"
            group = None
        else:
            group = color_groups[i]
            color = "rgba" + str(colorscale(groupsdict[group]))
            if group in groups:
                show_legend = True
                groups.remove(group)
            group = str(group)
        fig.add_scatter(
            x=axis_positions,
            y=y,
            line={"color": color},
            name=group,
            legendgroup=group,
            showlegend=show_legend,
        )
    # Axis lines
    for i, col_name in enumerate(column_names):
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
            y=[1.05],
            text=col_name,
            mode="text",
            showlegend=False,
        )
    return fig


def group_objectives(data):
    corr = spearmanr(data).correlation
    col_names = data.columns
    rank_list = []
    list_objs = []
    p = np.zeros_like(col_names)  # Number of partners
    group_id = np.zeros_like(col_names)  # Number of groups
    group_id[:] = np.inf
    groups = []

    for j in range(len(col_names) - 1):
        for i in range(j + 1, len(col_names)):
            rank_list.append(corr[i, j])
            list_objs.append((i, j))
    order = np.argsort(np.abs(rank_list))

    for i in order:
        obj1, obj2 = list_objs[i]
        if p[obj1] == 0 and p[obj2] == 0:
            # Case 1
            groups.append([obj1, obj2])
            group_id[[obj1, obj2]] = len(groups) - 1
            p[obj1] += 1
            p[obj2] += 1
        elif p[obj1] == 1 and p[obj2] == 0:
            # Case 2
            obj1_group = group_id[obj1]
            if groups[obj1_group].index(obj1) == 0:
                groups[obj1_group] = [obj2] + groups[obj1_group]
            elif groups[obj1_group].index(obj1) == len(groups[obj1_group]) - 1:
                groups[obj1_group] = groups[obj1_group] + [obj2]
            else:
                raise ValueError
            group_id[obj2] = obj1_group
            p[obj1] += 1
            p[obj2] += 1
        elif p[obj1] == 0 and p[obj2] == 1:
            # Case 3
            obj2_group = group_id[obj2]
            if groups[obj2_group].index(obj2) == 0:
                groups[obj2_group] = [obj1] + groups[obj2_group]
            elif groups[obj2_group].index(obj2) == len(groups[obj2_group]) - 1:
                groups[obj2_group] = groups[obj2_group] + [obj1]
            else:
                raise ValueError
            group_id[obj1] = obj2_group
            p[obj1] += 1
            p[obj2] += 1
        elif p[obj1] == 1 and p[obj2] == 1:
            # Case 4
            obj1_group = group_id[obj1]
            obj2_group = group_id[obj2]
            if obj1_group == obj2_group:
                continue
            obj1_group = groups[obj1_group]
            obj2_group = groups[obj2_group]
            groups[group_id[obj1]] = []
            groups[group_id[obj2]] = []
            if obj1_group.index(obj1) == 0 and obj2_group.index(obj2) == 0:
                obj1_group.reverse()
            if (
                obj2_group.index(obj2) == len(obj2_group) - 1
                and obj1_group.index(obj1) == len(obj1_group) - 1
            ):
                obj1_group.reverse()
            if (
                obj1_group.index(obj1) == 0
                and obj2_group.index(obj2) == len(obj2_group) - 1
            ):
                new_group = obj2_group + obj1_group
            elif (
                obj2_group.index(obj2) == 0
                and obj1_group.index(obj1) == len(obj1_group) - 1
            ):
                new_group = obj1_group + obj2_group
            else:
                print("hi")
                raise ValueError
            groups.append(new_group)
            group_id[new_group] = len(groups) - 1
            p[obj1] += 1
            p[obj2] += 1
        else:
            continue
    return groups[-1]


if __name__ == "__main__":
    data = pd.read_csv("c432-110.csv", header=None, sep="\s+")[range(2, 11)]
    # groups = KMeans(n_clusters=3).fit(data).labels_
    groups = AffinityPropagation().fit(data).labels_
    print(groups)
    obj_order = group_objectives(data)
    positions = np.linspace(0, 1, len(data.columns))
    pos = a = np.ones_like(obj_order, dtype=float)
    col_order = []
    for i, val in zip(obj_order, positions):
        pos[i] = val
        col_order.append(data.columns[i])
    print(col_order)
    data = data[col_order]
    parallel_coordinates(data, color_groups=groups,).show()
