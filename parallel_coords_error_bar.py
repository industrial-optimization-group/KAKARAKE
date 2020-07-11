import dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import json
import numpy as np
from functools import reduce
from dash.exceptions import PreventUpdate

names = ["x1", "x2", "x3","x4","x5"]  # Objective names
stdnames = [name + "_std" for name in names]
means = pd.DataFrame(np.random.rand(10, 5), columns=names)  # Mean values
std = pd.DataFrame(np.random.rand(10, 5) * 0.1, columns=stdnames)  # Standard deviation

df = means.join(std)


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            id="datatable-interactivity-container",
            children=[
                html.Div(id="brushes", children=[[]], hidden=True),
                dcc.Graph(id="par_coords", config={}),
            ],
        ),
        dash_table.DataTable(
            id="datatable-interactivity",
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True}
                for i in df.columns
            ],
            data=df.to_dict("records"),
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10,
        ),
    ]
)


@app.callback(
    Output("datatable-interactivity", "style_data_conditional"),
    [Input("datatable-interactivity", "selected_columns")],
)
def update_styles(selected_columns):
    return [
        {"if": {"column_id": i}, "background_color": "#D2F3FF"}
        for i in selected_columns
    ]


"""@app.callback(
    Output("par_coords", "figure"),
    [
        Input("datatable-interactivity", "derived_virtual_data"),
        Input("datatable-interactivity", "derived_virtual_selected_rows"),
    ],
)
def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = [
        "firebrick" if i in derived_virtual_selected_rows else "#A9A9A9"
        for i in range(len(dff))
    ]
    opacity = [
        1 if i in derived_virtual_selected_rows else 0.3 for i in range(len(dff))
    ]
    fig = go.Figure()
    for i in range(len(dff)):
        fig.add_scatter(
            x=[1, 2, 3],
            y=dff[names].loc[i].values,
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=dff[stdnames].loc[i].values,
                visible=True,
            ),
            line=dict(color=colors[i]),
            opacity=opacity[i],
        )

    return fig
"""


@app.callback(
    [Output("par_coords", "figure"), Output("brushes", "children")],
    [Input("par_coords", "selectedData")],
    [State("brushes", "children"), State("par_coords", "figure")],
)
def brush_plot(selectedData, brushes, fig):
    if selectedData is None or fig is None:
        fig = go.Figure(layout={"dragmode": "select"})
        for i in range(len(df)):
            fig.add_scatter(
                x=list(range(int(df.shape[1]/2))),
                y=df[names].loc[i].values,
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates
                    array=df[stdnames].loc[i].values,
                    visible=True,
                ),
                line=dict(color="red"), 
                opacity=1,
            )
        return [fig, brushes]
    else:
        # print(selectedData["range"])
        print(selectedData["points"])
        if not selectedData["points"]:
            raise PreventUpdate
        selected_indices = [dicts["curveNumber"] for dicts in selectedData["points"]]
        if not brushes[-1]:
            current_brush = list(range(len(fig["data"])))
            current_brush = [current_brush] * len(fig["data"][0]["x"])
            brushes = [current_brush]
        current_key = selectedData["points"][0]["pointNumber"]
        brushes[-1][current_key] = selected_indices
        selected_indices = reduce(lambda x, y: set(x) & set(y), brushes[-1])
        for index in range(len(fig["data"])):
            fig["data"][index]["line"]["color"] = (
                "red" if index in selected_indices else "grey"
            )
        return [fig, brushes]


if __name__ == "__main__":
    app.run_server(debug=True)
