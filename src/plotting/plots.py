from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

# Importing modules
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# importing processed data
data = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")

# Generating histogram plot with for numerical variables with Plotly Graph Objects
# Variables
hist_features = ['Age', 'Height', 'Weight', 'BMI', 'Systolic', 'Diastolic']
colour = 'DarkOrange'

# Initialize subplots
fig1 = make_subplots(rows=2, cols=1, row_heights=[0.2, 0.8], shared_xaxes=True)

# Add Traces with visibility settings
for idx, feature in enumerate(hist_features):
    fig1.add_trace(go.Histogram(
        x=data[feature],
        nbinsx=len(np.histogram_bin_edges(data[feature], bins='fd')),
        name=feature,
        marker_color=colour,
        visible=(idx == 0)
    ), row=2, col=1)
    fig1.add_trace(go.Box(
        x=data[feature],
        marker_symbol='line-ns-open',
        boxpoints='all',
        jitter=0,
        hoveron='points',
        name=feature,
        marker_color=colour,
        visible=(idx == 0)
    ), row=1, col=1)


# Add buttons
fig1.update_layout(
    updatemenus=[
        dict(
            direction="down",
            showactive=True,
            x=-0.2,
            xanchor='left',
            y=0.9,
            yanchor='top',
            buttons=list([
                dict(label=feature,
                     method="update",
                     args=[{"visible": [(i // 2 == idx) for i in range(len(hist_features) * 2)]},
                          {"xaxis2.title": feature}])
                for idx, feature in enumerate(hist_features)
            ]),
        )
    ],
    showlegend=False,
    title=dict(text='Histograms', x=0.01),
    yaxis1_title="",
    xaxis2_title=hist_features[0], # Default x-axis title for the first histogram
    yaxis2_title='Count', # y-axis title
)

# Add annotation
fig1.update_layout(
    annotations=[
        dict(text="Feature:", showarrow=False,
        x = -0.2, xref="paper", y=1, yref="paper", align="left")
    ]
)

# Set plot size and add a bar gap
fig1.update_layout(
    autosize=False,
    width=1100,
    height=400,
    margin=dict(l=20, r=20, t=50, b=20),
    bargap=0.15
)


fig1.show()

# Save the plot as an HTML file
fig1.write_html(FIGURES_DIR / "histogram.html")


# Generating barplot for selected categorical variables and participant age
# Variables
x_vars = ['uACR', 'BP_Category', 'S_Ethnicity']
y_var = 'Age'
k_var = ['BMI_Category', 'BP_Category', 'S_Ethnicity', 'Ethnicity_Black', 'Gender']
colour_list = ['DarkOrange', 'Sienna', 'Chocolate', 'DarkSalmon', 'Coral', 'SandyBrown']

# Initial x_var
current_x_var = x_vars[0]

fig2 = go.Figure()

# Add traces for each combination of k_var values
for a in k_var:
    for col, k in enumerate(data[a].unique()):
        fig2.add_trace(
            go.Bar(
                name=f'{k}',
                x=data[current_x_var].unique(),
                y=[data[y_var][(data[current_x_var] == x) & (data[a] == k)].mean() for x in data[current_x_var].unique()],
                marker_color=colour_list[col],
                visible=a == k_var[0],
            )
        )

# Create buttons for each k_var
buttons_k_var = []
for a in k_var:
    buttons_k_var.append(dict(
        method='update',
        label=a,
        args=[{
            'visible': [a == current for current in k_var for _ in data[current].unique()],
            'title.text': f'Mean Age by {a} and {current_x_var}'
        }]
    ))

# Create buttons for each x_var
buttons_x_var = []
for x in x_vars:
    buttons_x_var.append(dict(
        method='update',
        label=x,
        args=[{
            'x': [data[x].unique()] * len(k_var) * data[k_var[0]].nunique(),
            'y': [[data[y_var][(data[x] == val) & (data[a] == k)].mean() for val in data[x].unique()] for a in k_var for k in data[a].unique()],
            'title.text': f'Mean Age by {k_var[0]} and {x}'
        }]
    ))

# Update layout with dropdowns
fig2.update_layout(
    updatemenus=[
        dict(
            buttons=buttons_x_var,
            direction='down',
            showactive=True,
            x=-0.3,
            xanchor='left',
            y=0.9,
            yanchor='top'
        ),
        dict(
            buttons=buttons_k_var,
            direction='down',
            showactive=True,
            x=-0.3,
            xanchor='left',
            y=0.6,
            yanchor='top'
        )
    ],
    barmode='group',
    title=dict(text='Bivariate Analysis (Age)', x=0.01),
    yaxis_title='Mean Age',
    autosize=False,
    width=1100,
    height=350,
    margin=dict(l=20, r=20, t=50, b=20)
)

# Add annotation
fig2.update_layout(
    annotations=[
        dict(text="Select x Variable:", showarrow=False,
             x=-0.3, xref="paper", y=1, yref="paper", align="left"),
        dict(text="Select Feature:", showarrow=False,
             x=-0.3, xref="paper", y=0.7, yref="paper", align="left")
    ]
)

fig2.show()

# Save the plot as an HTML file
fig2.write_html(FIGURES_DIR / "categories_by_age.html")


# Generating barplot for selected categorical variables and participant pulse pressure
# Variables
x_vars = ['uACR', 'BMI_Category', 'BP_Category', 'Ethnicity_Black', 'Gender']
y_var = 'Pulse_Pressure'
k_var = ['BMI_Category', 'BP_Category', 'S_Ethnicity', 'Ethnicity_Black', 'Gender', 'uACR']
colour_list = ['DarkOrange', 'Sienna', 'Chocolate', 'DarkSalmon', 'Coral', 'SandyBrown']

# Initial x_var
current_x_var = x_vars[0]

fig3 = go.Figure()

# Add traces for each combination of k_var values
for a in k_var:
    for col, k in enumerate(data[a].unique()):
        fig3.add_trace(
            go.Bar(
                name=f'{k}',
                x=data[current_x_var].unique(),
                y=[data[y_var][(data[current_x_var] == x) & (data[a] == k)].mean() for x in data[current_x_var].unique()],
                marker_color=colour_list[col],
                visible=a == k_var[0],
            )
        )

# Create buttons for each k_var
buttons_k_var = []
for a in k_var:
    buttons_k_var.append(dict(
        method='update',
        label=a,
        args=[{
            'visible': [a == current for current in k_var for _ in data[current].unique()],
            'title.text': f'Mean Age by {a} and {current_x_var}'
        }]
    ))

# Create buttons for each x_var
buttons_x_var = []
for x in x_vars:
    buttons_x_var.append(dict(
        method='update',
        label=x,
        args=[{
            'x': [data[x].unique()] * len(k_var) * data[k_var[0]].nunique(),
            'y': [[data[y_var][(data[x] == val) & (data[a] == k)].mean() for val in data[x].unique()] for a in k_var for k in data[a].unique()],
            'title.text': f'Mean Pulse Pressure by {k_var[0]} and {x}'
        }]
    ))

# Update layout with dropdowns
fig3.update_layout(
    updatemenus=[
        dict(
            buttons=buttons_x_var,
            direction='down',
            showactive=True,
            x=-0.3,
            xanchor='left',
            y=0.9,
            yanchor='top'
        ),
        dict(
            buttons=buttons_k_var,
            direction='down',
            showactive=True,
            x=-0.3,
            xanchor='left',
            y=0.6,
            yanchor='top'
        )
    ],
    barmode='group',
    title=dict(text='Bivariate Analysis (Pulse Pressure)', x=0.01),
    yaxis_title='Mean Pulse Pressure',
    autosize=False,
    width=1100,
    height=350,
    margin=dict(l=20, r=20, t=50, b=20)
)

# Add annotation
fig3.update_layout(
    annotations=[
        dict(text="Select x Variable:", showarrow=False,
             x=-0.3, xref="paper", y=1, yref="paper", align="left"),
        dict(text="Select Feature:", showarrow=False,
             x=-0.3, xref="paper", y=0.7, yref="paper", align="left")
    ]
)

fig3.show()

# Save the plot as an HTML file
fig3.write_html(FIGURES_DIR / "categories_by_pulsepressure.html")

