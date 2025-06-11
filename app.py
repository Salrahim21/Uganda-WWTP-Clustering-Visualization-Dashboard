import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans

# Load and preprocess data
wwtp = pd.read_csv('Data/wwtp_input_data_uganda_new.csv')
df_cluster = wwtp[['treatment_type', 'Subregion', 'Capacity', 'Lat', 'Lon']].dropna()

df = df_cluster.copy()
df["AdjustedCapacity"] = np.clip(df["Capacity"], a_min=5, a_max=None)

kmeans = KMeans(n_clusters=3)
df["cluster"] = kmeans.fit_predict(df[["Lat", "Lon"]])

# Unique filter options
clusters = sorted(df["cluster"].unique())
treatment_types = sorted(df["treatment_type"].dropna().unique())

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "WWTP Clusters in Uganda"

# App layout
app.layout = html.Div([
    html.H1("WWTP Clusters in Uganda", style={'textAlign': 'center', 'color': '#ffffff'}),

    html.Div([
        html.Label("Filter by Cluster:", style={"color": "white"}),
        dcc.Dropdown(
            id='cluster-filter',
            options=[{"label": f"Cluster {c}", "value": c} for c in clusters],
            value=None,
            multi=True,
            placeholder="Select clusters"
        ),

        html.Label("Filter by Treatment Type:", style={"marginTop": 20, "color": "white"}),
        dcc.Dropdown(
            id='treatment-filter',
            options=[{"label": t, "value": t} for t in treatment_types],
            value=None,
            multi=True,
            placeholder="Select treatment types"
        ),
    ], style={"width": "40%", "display": "inline-block", "padding": "20px", "verticalAlign": "top"}),

    html.Div([
        dcc.Graph(id='map-graph', style={"height": "60vh", "width": "100%"}),
    ], style={"width": "100%", "display": "inline-block"}),

    html.Div([
        html.H3("Total Capacity by Treatment Type", style={"color": "white"}),
        dcc.Graph(id='bar-chart')
    ], style={"padding": "20px"}),

    html.Div([
        html.H3("Correlation Heatmap", style={"color": "white"}),
        html.Img(id='heatmap-img')
    ], style={"padding": "20px", "textAlign": "center"}),

], style={"backgroundColor": "#111111", "padding": "10px"})


# Callback for all visuals
@app.callback(
    Output('map-graph', 'figure'),
    Output('bar-chart', 'figure'),
    Output('heatmap-img', 'src'),
    Input('cluster-filter', 'value'),
    Input('treatment-filter', 'value')
)
def update_visuals(selected_clusters, selected_treatments):
    filtered_df = df.copy()

    if selected_clusters:
        filtered_df = filtered_df[filtered_df['cluster'].isin(selected_clusters)]

    if selected_treatments:
        filtered_df = filtered_df[filtered_df['treatment_type'].isin(selected_treatments)]

    # Map
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="Lat",
        lon="Lon",
        color="cluster",
        size="AdjustedCapacity",
        size_max=30,
        hover_name="Subregion",
        text="Subregion",
        hover_data={"Capacity": True, "treatment_type": True, "cluster": True},
        zoom=6,
        height=600
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        paper_bgcolor="#111111",
        font_color="white"
    )

    # Bar Chart
    bar_fig = px.bar(
        filtered_df.groupby("treatment_type")["Capacity"].sum().reset_index(),
        x="treatment_type",
        y="Capacity",
        title="Total Capacity by Treatment Type",
        labels={"Capacity": "Total Capacity", "treatment_type": "Treatment Type"},
        color="treatment_type"
    )
    bar_fig.update_layout(paper_bgcolor="#111111", font_color="white")

    # Correlation Heatmap as base64 image
    corr = filtered_df[["Capacity", "AdjustedCapacity"]].corr()
    plt.figure(figsize=(4, 3))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    data_uri = base64.b64encode(buf.getbuffer()).decode("utf8")
    heatmap_src = "data:image/png;base64," + data_uri

    return fig_map, bar_fig, heatmap_src


if __name__ == '__main__':
    app.run(debug=True)
