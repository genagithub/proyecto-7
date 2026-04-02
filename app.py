import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State 

df = pd.read_csv("data/properati.csv")

df["lat"], df["lon"] = df["lon"], df["lat"]

df.dropna(subset=["lat","lon","surface_covered","price","currency"], inplace=True)
df = df[df["surface_covered"] <= df["surface_total"]]

cols_mode = ["rooms","bedrooms","bathrooms"]

for col in cols_mode:
    if col in df.columns:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)
        
imputer = KNNImputer(n_neighbors=5)
cols_knn = ["lat","lon","rooms","bathrooms","bedrooms","surface_covered","price","surface_total"]

imputed_array = imputer.fit_transform(df[cols_knn])
df["surface_total"] = imputed_array[:, -1]

def get_dolar():
    try:
        res = requests.get("https://dolarapi.com/v1/dolares/blue", timeout=3)
        return float(res.json()["venta"])
    except:
        return 1360.0

dolar_value = get_dolar()
currency_ARS = df.loc[df["currency"] == "ARS", "price"]
df.loc[df["currency"] == "ARS", "price"] = currency_ARS / dolar_value

df["end_date"] = df["end_date"].astype(str).mask(df["end_date"].astype(str).str.startswith("9999", na=False), np.nan)

df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

today = pd.to_datetime("today")
publicate_days = (df["end_date"].fillna(today) - df["start_date"]).dt.days
status = np.where(df["end_date"].isna(), "Activo", "Finalizado")

df.insert(2, "publicate_days", publicate_days)
df.insert(3, "status", status)

df = df.drop(["start_date","end_date","created_on","currency"], axis=1)

df.loc[df["operation_type"] == "Venta", "price_period"] = "Pago único"

df_sales = df[df["operation_type"] == "Venta"]
df_rents = df[df["operation_type"] == "Alquiler"]
df_temp_rents = df[df["operation_type"] == "Alquiler temporal"]

rent_mode = df_rents["price_period"].mode()[0]
df_rents["price_period"] = df_rents["price_period"].fillna(rent_mode)
temp_rent_mode = df_temp_rents["price_period"].mode()[0]
df_temp_rents["price_period"] = df_temp_rents["price_period"].fillna(temp_rent_mode)

df.loc[df["operation_type"] == "Alquiler", "price_period"] = df_rents["price_period"]
df.loc[df["operation_type"] == "Alquiler temporal", "price_period"] = df_temp_rents["price_period"]

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(id="body",className="e7_body",children=[
        html.A(href="https://github.com/genagithub/proyecto-7/blob/main/estimación_y_agrupamiento_de_precios_inmuebles.ipynb",children=[html.H1("Análisis inmobiliario de CABA",id="title",className="e7_title")]),
        html.Div(id="div_dropdown",className="e7_div_dropdown",children=[
            dcc.Dropdown(id="dropdown_1",className="e7_dropdown",
                        options=df["operation_type"].unique(),
                        value=df["operation_type"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_2",className="e7_dropdown",
                        options=df["price_period"].unique(),
                        value=df["price_period"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_3",className="e7_dropdown",
                        options=df["status"].unique(),
                        value=df["status"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_4",className="e7_dropdown",
                        options=df["property_type"].unique(),
                        value=df["property_type"].unique()[0],
                        multi=False,
                        clearable=False),
        ]),
        dcc.Graph(id="graph_1",className="e7_graph",figure={}),
        dcc.Graph(id="graph_2",className="e7_graph",figure={})
])

@app.callback(
    [Output(component_id="graph_1",component_property="figure"),
    Output(component_id="graph_2",component_property="figure"),
    Output(component_id="dropdown_2",component_property="value")],
    [Input(component_id="dropdown_1",component_property="value"),
    State(component_id="dropdown_2",component_property="value"),
    Input(component_id="dropdown_3",component_property="value"),
    Input(component_id="dropdown_4",component_property="value")]
)

def update_graph(slct_operation, slct_price_period, slct_status, slct_property):     
    if slct_operation in ["Alquiler", "Alquiler temporal"]:
        slct_price_period = "Mensual"
        price = df_filtered["price"]
    elif slct_operation == "Venta":
        slct_price_period = "Pago único"
        price = np.log1p(df_filtered["price"])    

    df_filtered = df.loc[(df["operation_type"] == slct_operation) & (df["price_period"] == slct_price_period) & (df["status"] == slct_status) & (df["property_type"] == slct_property), :]
    df_filtered["lat"] = pd.to_numeric(df_filtered["lat"])
    df_filtered["lon"] = pd.to_numeric(df_filtered["lon"])  

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
        margin={"r":0,"t":0,"l":0,"b":0},
    )
    
    eps_config = {
        "Venta": 0.2,             
        "Alquiler": 0.3,          
        "Alquiler temporal": 0.15 
    }
    current_eps = eps_config.get(slct_operation, 0.3)
    
    scaler = RobustScaler()
    scaled_price = scaler.fit_transform(price.values.reshape(-1, 1))
    
    dbscan = DBSCAN(eps=current_eps, min_samples=5) 
    df_filtered["clusters"] = dbscan.fit_predict(scaled_price)

    real_clusters = df_filtered[df_filtered["clusters"] != -1]
    cluster_stats = real_clusters.groupby("clusters")["price"].agg(["min", "max"]).sort_values("min")
    
    label_map = {}
    for i, (idx, row) in enumerate(cluster_stats.iterrows(), start=1):
        label_map[idx] = f"Rango {i}: (${int(row["min"])} - ${int(row["max"])})"

    label_map[-1] = "Precio Atípico / Ruido"
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
    
    clusters_analysis.update_layout(height=850, template="plotly_dark")
    
    return caba_map, clusters_analysis, slct_price_period
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
