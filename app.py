import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
from dash.dependencies import Input, Output


data = datasets.fetch_california_housing()

df = pd.DataFrame(data["data"],columns=data["feature_names"])
df["MedHouseVal"] = data["target"]
df["MedHouseVal"] = df["MedHouseVal"] * 100000

x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]],
                                                    df["MedHouseVal"],
                                                    test_size=0.25)

turned_parameters = {
    "n_estimators":[100,200,300,400,500],
    "subsample":[0.7,0.75,0.8,0.85,0.9],
    "max_depth":[3,4,5,6,7],
    "learning_rate":[0.2,0.3,0.4,0.5,0.55],
    "min_child_weight":[2,3,4,5,6],
    "gamma":[0,1,2,3,4]
}

xgbr_test = XGBRegressor()

random_search = RandomizedSearchCV(xgbr_test, turned_parameters,cv=5)
random_search.fit(df[df.columns[:-1]], df["MedHouseVal"])

xgbr = XGBRegressor(n_estimators = random_search.best_params_["n_estimators"],
                    subsample = random_search.best_params_["subsample"],
                    max_depth = random_search.best_params_["max_depth"],
                    learning_rate = random_search.best_params_["learning_rate"],
                    min_child_weight = random_search.best_params_["min_child_weight"],
                    gamma = random_search.best_params_["gamma"])

xgbr.fit(x_train, y_train)


kmeans = KMeans(n_clusters=5).fit(df["MedHouseVal"].values.reshape((-1,1)))

clusters = kmeans.labels_

df["clusters"] = clusters

range_values = np.array([])

for c in df["clusters"].sort_values().unique():
    cluster = df.loc[df["clusters"] == c,["clusters","MedHouseVal"]]
    max_value = str(cluster["MedHouseVal"].max())
    min_value = str(cluster["MedHouseVal"].min())
    range_values = np.append(range_values,min_value)
    range_values = np.append(range_values,max_value)
    
range_values = range_values.reshape((-1,2))
    
df["clusters"] = df["clusters"].replace(
    {
        0:f"0 ({range_values[0,0][:8]}$-{range_values[0,1][:8]}$)",
        1:f"1 ({range_values[1,0][:8]}$-{range_values[1,1][:8]}$)",
        2:f"2 ({range_values[2,0][:8]}$-{range_values[2,1][:8]}$)",
        3:f"3 ({range_values[3,0][:8]}$-{range_values[3,1][:8]}$)",
        4:f"4 ({range_values[4,0][:8]}$-{range_values[4,1][:8]}$)"
    })

clusters_count = df["clusters"].value_counts().reset_index()

cluster_map = make_subplots(rows=2, cols=1, subplot_titles=["Rangos de tasación","Conteo de rangos"])

def make_figure(df, cluster):

    var_x = df.loc[df["clusters"] == cluster,["Longitude", "clusters"]]
    var_y = df.loc[df["clusters"] == cluster,["Latitude", "clusters"]]

    cluster_map.add_trace(go.Scatter( 
        x=var_x["Longitude"], 
        y=var_y["Latitude"], 
        mode="markers", 
        marker=dict( 
        size=9, 
        symbol="circle", 
        line=dict(width=0.5, color="white") 
        ), 
        name=cluster), row=1, col=1) 

make_figure(df, df["clusters"].unique()[0])
make_figure(df, df["clusters"].unique()[1])
make_figure(df, df["clusters"].unique()[2])
make_figure(df, df["clusters"].unique()[3])
make_figure(df, df["clusters"].unique()[4])

cluster_map.update_xaxes(row=1, col=1, range=[-125,-114], constrain="domain")
cluster_map.update_yaxes(row=1, col=1, range=[32,42], constrain="domain", scaleanchor="x", scaleratio=1)

clusters_count = df["clusters"].value_counts().reset_index()

cluster_map.add_trace(go.Bar(x=clusters_count["clusters"], y=clusters_count["count"]), row=2, col=1)
cluster_map.update_layout(height=900)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(id="body",className="e7_body",children=[
        html.H1("Análsis inmobiliario en California ",id="title",className="e7_title",href="https://github.com/genagithub/proyecto-7/blob/main/estimaci%C3%B3n_de_valores_inmobiliarios_y_segmentaci%C3%B3n_de_rangos.ipynb",target="blank"),
        html.Div(id="div_dropdown",className="e7_div_dropdown",children=[
            dcc.Dropdown(id="dropdown",className="e7_dropdown",
                        options = [
                            {"label":"Valor de precio","value":"MedHouseVal"},
                            {"label":"Ingreso medio","value":"MedInc"},
                            {"label":"Edad media","value":"HouseAge"},
                            {"label":"Promedio de habitaciones","value":"AveRooms"},
                            {"label":"Promedio de dormitorios","value":"AveBedrms"},
                            {"label":"Población","value":"Population"},
                            {"label":"Promedio de ocupación","value":"AveOccuption"}
                        ],
                        value="MedHouseVal",
                        multi=False,
                        clearable=False)
        ]),
        html.Div(id="div_graphs",className="e7_div",children=[
        dcc.Graph(id="graph_1",className="e7_graph",figure={}),
        dcc.Graph(id="graph_2",className="e7_graph",figure=cluster_map)
        ])
])

@app.callback(
    Output(component_id="graph_1",component_property="figure"),
    [Input(component_id="dropdown",component_property="value")]
)

def update_graph(slct_var):
    
    california_map = go.Figure(go.Scattermapbox(
        lat=df["Latitude"],
        lon=df["Longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=9,
            color=df[slct_var],
            cmin=df[slct_var].min(),
            cmax=df[slct_var].max(),
            showscale=True
        )
    ))
    
    california_map.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=4.8,
    mapbox_center_lat = 37.0,
    mapbox_center_lon = -119.0,
    margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    return california_map
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
