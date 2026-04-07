import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

df = pd.read_csv("data/properati_processed.csv", usecols=["operation_type", "price_period", "status", "property_type", "price", "lat", "lon"])

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(id="body",className="e7_body",children=[
        html.A(href="https://github.com/genagithub/proyecto-7/blob/main/estimación_y_agrupamiento_de_precios_inmuebles.ipynb",children=[html.H1("Análisis inmobiliario de CABA",id="title",className="e7_title")]),
        html.Div(id="div_dropdown",className="e7_div_dropdown",children=[
            dcc.Dropdown(id="dropdown_1",className="e7_dropdown",
                        options=df["operation_type"].unique().tolist(),
                        value=df["operation_type"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_2",className="e7_dropdown",
                        options=df["price_period"].unique().tolist(),
                        value=df["price_period"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_3",className="e7_dropdown",
                        options=df["status"].unique().tolist(),
                        value=df["status"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_4",className="e7_dropdown",
                        options=df["property_type"].unique().tolist(),
                        value=df["property_type"].unique()[0],
                        multi=False,
                        clearable=False),
        ]),
        html.Div(id="div_graph_1",className="e7_div_graph",children=[
            dcc.Graph(id="graph_1",className="e7_graph",figure={})   
        ]),
        html.Div(id="div_graph_2",className="e7_div_graph",style={"margin-bottom":"20px"},children=[
            dcc.Graph(id="graph_2",className="e7_graph",figure={})    
        ])
])

@app.callback(
    [Output(component_id="graph_1",component_property="figure"),
    Output(component_id="graph_2",component_property="figure")],
    [Input(component_id="dropdown_1",component_property="value"),
    Input(component_id="dropdown_2",component_property="value"),
    Input(component_id="dropdown_3",component_property="value"),
    Input(component_id="dropdown_4",component_property="value")]
)

def update_graph(slct_operation, slct_price_period, slct_status, slct_property):     
    df_filtered = df.loc[(df["operation_type"] == slct_operation) & (df["price_period"] == slct_price_period) & (df["status"] == slct_status) & (df["property_type"] == slct_property), :].copy()
    df_filtered.dropna(inplace=True)
        
    if df_filtered.empty or len(df_filtered) < 5:
        fig_empty = go.Figure().update_layout(title="Sin datos suficientes para esta selección", template="plotly_dark")
        return fig_empty, fig_empty
    
    df_filtered["lat"] = pd.to_numeric(df_filtered["lat"], errors="coerce")
    df_filtered["lon"] = pd.to_numeric(df_filtered["lon"], errors="coerce")  

    caba_map = go.Figure(go.Scattermapbox(
        lat=df_filtered["lat"],
        lon=df_filtered["lon"],
        mode="markers",
        text=df_filtered["price"], 
        hovertemplate="Precio: USD %{text:,.0f}<extra></extra>",
        marker=go.scattermapbox.Marker(
            size=8,
            color=df_filtered["price"],
            cmin=df_filtered["price"].min(),
            cmax=df_filtered["price"].max(),
            showscale=True,
            colorbar=dict(title="Precios")
        )
    ))
    
    caba_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=11.5,
        mapbox_center={"lat": -34.6037, "lon": -58.4417},
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    prices_filter = {
        "Venta": (10000, 10000000),
        "Alquiler": (150, 15000),
        "Alquiler temporal": (200, 20000)
    }

    min_p, max_p = prices_filter.get(slct_operation, (10000, 10000000))
    df_filtered = df_filtered[df_filtered["price"].between(min_p, max_p)].copy()

    price_data = np.log1p(df_filtered["price"]) if slct_operation == "Venta" else df_filtered["price"]
    scaled_price = RobustScaler().fit_transform(price_data.values.reshape(-1, 1))

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df_filtered["clusters"] = kmeans.fit_predict(scaled_price)

    cluster_stats = df_filtered.groupby("clusters")["price"].agg(["min", "max", "count"]).sort_values("min")

    label_map = {
        idx: f"Rango {i+1}: (${int(row['min']):,} - ${int(row['max']):,})" 
        for i, (idx, row) in enumerate(cluster_stats.iterrows())
    }

    df_filtered["cluster_label"] = df_filtered["clusters"].map(label_map)

    clusters_analysis = make_subplots(
        rows=2, cols=1, 
        subplot_titles=["Distribución Geográfica por Rango", "Cantidad de Propiedades"],
        vertical_spacing=0.1
    )

    for label in sorted(df_filtered["cluster_label"].unique()):
        df_c = df_filtered[df_filtered["cluster_label"] == label]
        clusters_analysis.add_trace(
            go.Scatter(
                x=df_c["lon"], 
                y=df_c["lat"], 
                mode="markers",
                name=label,
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>Lat: %{{y}}<br>Lon: %{{x}}<extra></extra>"
            ), row=1, col=1
        )

    clusters_analysis.update_xaxes(range=[-58.55, -58.33], row=1, col=1)
    clusters_analysis.update_yaxes(range=[-34.72, -34.52], scaleanchor="x", scaleratio=1, row=1, col=1)
    
    counts = df_filtered["cluster_label"].value_counts().reset_index()
    clusters_analysis.add_trace(
        go.Bar(x=counts["cluster_label"], y=counts["count"], name="Cantidad"),
        row=2, col=1
    )
    
    clusters_analysis.update_layout(height=850, template="plotly_dark", margin=dict(t=50, l=25, r=25, b=25))
    
    return caba_map, clusters_analysis
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
