import os
import numpy as np
import pandas as pd

# import plotly.express as ex
# import plotly.graph_objects as go
from kakarake.parallel_coordinates_bands import (
    parallel_coordinates_bands_lines,
    GaussianMixtureclustering,
    DBSCANclustering,
    order_objectives,
    calculate_axes_positions,
)


from scipy.stats import pearsonr

from tsp_solver.greedy import solve_tsp

# All files
# files = [name.split(".csv")[:-1][0] for name in os.listdir("data")]
# Some files
# files = ["MetallurgicalProblem4d", "MetallurgicalProblem3d"]
num_files = len(files)


def auto_par_coords(
    data: pd.DataFrame,
    solutions: bool = True,
    bands: bool = True,
    medians: bool = True,
    dist_parameter: float = 0.05,
    use_absolute_corr: bool = False,
    distance_formula: int = 1,
    clustering: str = "DBSCAN",
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
    if clustering == "DBSCAN":
        groups = np.asarray(DBSCANclustering(ordered_data))
    elif clustering == "Gaussian":
        groups = np.asarray(GaussianMixtureclustering(ordered_data))
    groups = groups - np.min(groups)  # translate minimum to 0.
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

