from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from pages import home, form, findings
from app import app

# Sidebar layout
sidebar = dbc.Nav(
    [
        dbc.NavLink("Home", href="/", active="exact"),
        dbc.NavLink("Findings", href="/findings", active="exact"),
        dbc.NavLink("Predict Your CKD Risk", href="/form", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="bg-light",
)

app.layout = dbc.Container([ # type: ignore
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col(
            sidebar,
            width=2,
            className="d-flex flex-column flex-shrink-0 p-3 bg-light vh-100",
        ),
        dbc.Col(id="page-content", width=10, className="p-4"),
    ]),
], fluid=True)

@app.callback( # type: ignore
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/":
        return home.layout
    elif pathname == "/form":
        return form.layout
    elif pathname == "/findings":
        return findings.layout
    return html.P("404 - Page not found")

if __name__ == '__main__':
    app.run(debug=False)
