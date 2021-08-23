import os
import numpy as np
import pandas as pd

# import plotly.express as ex
# import plotly.graph_objects as go
from kakarake.parallel_coordinates_bands import (
    parallel_coordinates_bands_lines,
    GaussianMixtureclustering,
    DBSCANclustering,
)


from scipy.stats import pearsonr

from tsp_solver.greedy import solve_tsp


files = [name.split(".csv")[:-1][0] for name in os.listdir("data")]
num_files = len(files)


def auto_par_coords(
    data: pd.DataFrame,
    solutions: bool = True,
    bands: bool = True,
    medians: bool = True,
    dist_parameter: float = 0.05,
):
    # Calculating correlations
    corr = np.asarray(
        [
            [
                pearsonr(data.values[:, i], data.values[:, j])[0]
                for j in range(len(data.columns))
            ]
            for i in range(len(data.columns))
        ]
    )
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
    groups = np.asarray(DBSCANclustering(data))
    groups = groups - np.min(groups)
    fig1 = parallel_coordinates_bands_lines(
        data,
        color_groups=groups,
        axis_positions=axis_dist,
        solutions=solutions,
        bands=bands,
        medians=medians,
    )
    return fig1


i = 0
for file in files:
    print(f"file {i} of {num_files}")
    i += 1
    data = pd.read_csv("data/" + file + ".csv")
    try:
        auto_par_coords(data).write_html("images/" + file + ".html")
    except ValueError as err:
        print(f"Error in file: {file}")
        print(err)

