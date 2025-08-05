# index.py
from dash import dcc, html, Input, Output
from app import app
from pages import home, units_resold, median_price

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # Sidebar
    html.Div(className='sidebar', children=[
        html.H2("Navigation"),
        dcc.Link('Home', href='/'),
        html.Br(),
        dcc.Link('Units Resold', href='/units-resold'),
        html.Br(),
        dcc.Link('Median Price', href='/median-price'),
    ]),

    # Main content
    html.Div(id='page-content', className='content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/units-resold':
        return units_resold.layout
    elif pathname == '/median-price':
        return median_price.layout
    else:
        return home.layout  # default page

if __name__ == '__main__':
    app.run_server(debug=True)
    