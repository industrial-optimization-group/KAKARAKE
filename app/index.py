import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State

from kakarake.parallel_coordinates_bands import auto_par_coords
import pandas as pd
import base64
import io

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

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
                                        ],
                                        value=["solns", "bands"],
                                        inline=True,
                                    )
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
                                            className="ml-2",
                                        ),
                                    ],
                                ),
                            ],
                            inline=True,
                        )
                    ],
                    id="advanced-collapse",
                ),
                className="row justify-content-center mt-3 mb-3",
            )
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
                    dcc.Graph(
                        id="heatmap",
                        config=dict(displayModeBar=False, scrollZoom=False),
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


def parse_contents(filename, contents):
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

    return df


@app.callback(
    Output("parcoord", "figure"),
    Output("heatmap", "figure"),
    Output("filename", "children"),
    Input("upload-data", "filename"),
    State("upload-data", "contents"),
    State("advanced-checklist", "value"),
    State("dist-param", "value"),
    prevent_initial_call=True,
)
def update_output(filename, contents, checklist, dist_parameter):
    if filename is not None:
        solns = True if "solns" in checklist else False
        bands = True if "bands" in checklist else False
        medians = True if "medians" in checklist else False
        data = parse_contents(filename, contents)

        return (
            *auto_par_coords(
                data,
                solutions=solns,
                bands=bands,
                medians=medians,
                dist_parameter=dist_parameter,
            ),
            filename,
        )


if __name__ == "__main__":
    app.run_server(debug=True)
