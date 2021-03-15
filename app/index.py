import dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State

from kakarake.parallel_coordinates_bands import (
    parallel_coordinates_bands_lines,
    annotated_heatmap,
    order_objectives,
    calculate_axes_positions,
)
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
)
from kakarake.parallel_coords import GaussianMixtureclustering
import pandas as pd
import numpy as np
import base64
import io

from sklearn import manifold
from functools import partial
from plotly import express as ex
from sklearn.preprocessing import StandardScaler

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

# global vars
df = pd.DataFrame()
corr = []
obj_order = []
original_order = []
modified_df = []
axis_dist = []
axis_signs = []

n_neighbors = 10
n_components = 2

methods = {}

LLE = partial(
    manifold.LocallyLinearEmbedding, n_neighbors, n_components, eigen_solver="auto"
)

methods["LLE"] = LLE(method="standard")
methods["LTSA"] = LLE(method="ltsa")
methods["Hessian LLE"] = LLE(method="hessian")
methods["Modified LLE"] = LLE(method="modified")
methods["Isomap"] = manifold.Isomap(n_neighbors, n_components)
methods["MDS"] = manifold.MDS(n_components, max_iter=100, n_init=1)
methods["SE"] = manifold.SpectralEmbedding(
    n_components=n_components, n_neighbors=n_neighbors
)
methods["t-SNE"] = manifold.TSNE(n_components=n_components, init="pca", random_state=0)

app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Fluid Parallel Coordinates"),
                className="row justify-content-center",
            ),
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        dbc.Button("Input Data", color="primary",), id="upload-data"
                    ),
                    width={"size": 1, "offset": 4},
                ),
                dbc.Col(html.Div(children="", id="filename"), width={"size": 2}),
                dbc.Col(
                    dbc.Button(
                        "Advanced Options", id="advanced-options", color="primary"
                    ),
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                dbc.Collapse(
                    children=[
                        dbc.Form(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Number of solution clusters"),
                                        dbc.Input(
                                            id="num-soln-clusters",
                                            type="number",
                                            step=1,
                                            min=0,
                                            value=0,
                                            placeholder="Put '0' for automated clustering",
                                        ),
                                    ]
                                ),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Solution clustering algorithm"),
                                        dcc.Dropdown(
                                            options=[
                                                {"label": "K-Means", "value": "KM",},
                                                {
                                                    "label": "Spectral Clustering",
                                                    "value": "SC",
                                                },
                                                {
                                                    "label": "Ward Hierarchal Clustering",
                                                    "value": "WHC",
                                                },
                                                {
                                                    "label": "Agglomerative Clustering",
                                                    "value": "AC",
                                                },
                                                {
                                                    "label": "Gaussian Mixture",
                                                    "value": "GMM",
                                                },
                                            ],
                                            value="GMM",
                                            multi=False,
                                            id="soln-cluster-dropdown",
                                        ),
                                    ]
                                ),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Distance Parameter"),
                                        dbc.Input(
                                            id="dist-param",
                                            type="number",
                                            step=0.01,
                                            min=0,
                                            max=1,
                                            value=0.1,
                                        ),
                                    ],
                                ),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Dimension reduction algorithm"),
                                        dcc.Dropdown(
                                            options=[
                                                {"label": "LLE", "value": "LLE",},
                                                {"label": "LTSA", "value": "LTSA",},
                                                {
                                                    "label": "Hessian LLE",
                                                    "value": "Hessian LLE",
                                                },
                                                {
                                                    "label": "Modified LLE",
                                                    "value": "Modified LLE",
                                                },
                                                {"label": "Isomap", "value": "Isomap",},
                                                {"label": "MDS", "value": "MDS",},
                                                {"label": "SE", "value": "SE",},
                                                {"label": "t-SNE", "value": "t-SNE",},
                                            ],
                                            value="t-SNE",
                                            multi=False,
                                            id="reduced-plot-dropdown",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dbc.Form(
                            [
                                dbc.FormGroup(
                                    dbc.Checklist(
                                        id="advanced-checklist",
                                        options=[
                                            {
                                                "label": "Show Solutions",
                                                "value": "solns",
                                            },
                                            {"label": "Show Bands", "value": "bands"},
                                            {
                                                "label": "Show Medians",
                                                "value": "medians",
                                            },
                                            {
                                                "label": "Allow axes inversion",
                                                "value": "axes",
                                            },
                                            {
                                                "label": "Use absolute correlations",
                                                "value": "abs_corr",
                                            },
                                        ],
                                        value=["solns", "bands"],
                                        inline=True,
                                    )
                                )
                            ],
                            className="row justify-content-center mt-3 mb-3",
                        ),
                    ],
                    id="advanced-collapse",
                ),
                className="row justify-content-center mt-3 mb-3",
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Calculate axes positions", id="calc-axes-btn", color="primary"
                    ),
                    width={"size": 2, "offset": 2},
                    className="row justify-content-end",
                ),
                dbc.Col(
                    dbc.Button(
                        "Perform clustering", id="soln-clustering-btn", color="primary"
                    ),
                    width={"size": 2, "offset": 1},
                    className="row justify-content-center",
                ),
                dbc.Col(
                    dbc.Button("Plot data", id="plot-btn", color="primary"),
                    width={"size": 2, "offset": 1},
                ),
            ],
            className="mt-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="parcoord",
                        config=dict(displayModeBar=False, scrollZoom=False),
                        style={"height": "850px"},
                    ),
                    width={"size": 7, "offset": 1},
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            dcc.Graph(
                                id="heatmap",
                                config=dict(displayModeBar=False, scrollZoom=False),
                            ),
                            dcc.Graph(
                                id="reduced-plot",
                                config=dict(displayModeBar=False, scrollZoom=False),
                            ),
                        ]
                    ),
                    width={"size": 3},
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button("Export plot", color="primary"),
                className="row justify-content-center",
            )
        ),
        html.Div(children=[], id="calc-axes-dump", hidden=True),
        html.Div(children=[], id="soln-cluster-dump", hidden=True),
    ]
)


@app.callback(
    Output("advanced-collapse", "is_open"),
    [Input("advanced-options", "n_clicks")],
    [State("advanced-collapse", "is_open")],
    prevent_initial_call=True,
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("filename", "children"),
    Input("upload-data", "filename"),
    State("upload-data", "contents"),
    State("advanced-checklist", "value"),
    prevent_initial_call=True,
)
def parse_contents(filename, contents, checklist):
    global df
    global original_order
    if filename is None:
        PreventUpdate()
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return 0
    if "ralph" in filename:
        df = -df
    original_order = df.columns
    return filename


@app.callback(
    Output("calc-axes-dump", "children"),
    Input("calc-axes-btn", "n_clicks"),
    State("dist-param", "value"),
    State("advanced-checklist", "value"),
    prevent_initial_call=True,
)
def calculate_axes(button_click, distance_parameter, checklist):
    global df, obj_order, corr, axis_signs, axis_dist, modified_df

    use_absolute_corr = True if "abs_corr" in checklist else False
    corr, obj_order = order_objectives(df, use_absolute_corr)

    modified_df, axis_dist, axis_signs = calculate_axes_positions(
        df, obj_order, corr, distance_parameter
    )
    if "axes" not in checklist:
        axis_signs = list(np.abs(axis_signs))
    return 1


@app.callback(
    Output("soln-cluster-dump", "children"),
    Input("soln-clustering-btn", "n_clicks"),
    State("num-soln-clusters", "value"),
    State("soln-cluster-dropdown", "value"),
    prevent_initial_call=True,
)
def soln_clustering(button_click, num_clusters, clustering_algorithm):
    global modified_df
    if num_clusters == 0 or clustering_algorithm == "GMM":
        groups = GaussianMixtureclustering(modified_df)
    elif clustering_algorithm == "KM":
        groups = KMeans(n_clusters=num_clusters).fit(modified_df).labels_
    elif clustering_algorithm == "SC":
        groups = SpectralClustering(n_clusters=num_clusters).fit(modified_df).labels_
    elif clustering_algorithm == "WHC":
        groups = (
            AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
            .fit(modified_df)
            .labels_
        )
    elif clustering_algorithm == "AC":
        groups = (
            AgglomerativeClustering(n_clusters=num_clusters, linkage="single")
            .fit(modified_df)
            .labels_
        )
    return groups


@app.callback(
    Output("parcoord", "figure"),
    Output("heatmap", "figure"),
    Output("reduced-plot", "figure"),
    Input("plot-btn", "n_clicks"),
    State("advanced-checklist", "value"),
    State("dist-param", "value"),
    State("soln-cluster-dump", "children"),
    State("reduced-plot-dropdown", "value"),
    prevent_initial_call=True,
)
def update_output(button_click, checklist, dist_parameter, groups, red_dim_algo):
    global df, axis_dist, axis_signs, corr, original_order, obj_order, methods, LLE
    if not button_click:
        PreventUpdate()
    solns = True if "solns" in checklist else False
    bands = True if "bands" in checklist else False
    medians = True if "medians" in checklist else False

    reduced_data = StandardScaler().fit_transform(df)
    reduced_data = methods[red_dim_algo].fit_transform(reduced_data)
    fig = ex.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color=groups)
    fig.update_xaxes(zeroline=False, showticklabels=False, title_text="")
    fig.update_yaxes(zeroline=False, showticklabels=False, title_text="")
    fig.update_layout(title="Reduced view",)
    fig.layout.coloraxis.showscale = False
    return (
        parallel_coordinates_bands_lines(
            modified_df,
            color_groups=groups,
            axis_positions=axis_dist,
            axis_signs=axis_signs,
            solutions=solns,
            bands=bands,
            medians=medians,
        ),
        annotated_heatmap(corr, original_order, obj_order),
        fig,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
