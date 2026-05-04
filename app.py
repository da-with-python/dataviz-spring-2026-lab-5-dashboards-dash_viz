from flask import Flask, render_template
import dash
from dash import html, dcc, dash_table
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.colors as mcolors
from itertools import combinations, chain
import plotly.graph_objects as go
import re
from network import plot_communities, get_communities, create_network
from ast import literal_eval

# загрузка данных
kdf = pd.read_csv("titles_keywords.csv")

# подготовка данных для столбчатых диаграмм
df = kdf["title"].value_counts().reset_index()[:10]
analyst = kdf[kdf["title"].str.contains("Аналитик|analyst", flags=re.IGNORECASE)]["title"].reset_index()
df2 = analyst["title"].value_counts().reset_index()[:10]
fig = px.bar(df, x="title", y="count", title="Топ-10")
fig2 = px.bar(df2, x="title", y="count", title="Топ-10")

# подготовка данных для графа
kdf["keywords"] = kdf["keywords"].apply(literal_eval)
kdf = kdf[["title", "keywords"]]
net = create_network(kdf)
G = nx.Graph()
G.add_edges_from(net)
communities, S = get_communities(G)
fig3 = plot_communities(communities, S)

# подготовка данных для вывода таблицы
kdf["keywords"] = kdf["keywords"].apply(lambda x: ", ".join(x))


# стартовая страница
server = Flask(__name__)

@server.route('/')
def index():
    return render_template("index.html")


# страница с исходными данными
dash_dashboard_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/data/',
    suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(style={
    'fontFamily': 'Sans-serif',
    'textAlign': 'center',
    'padding': '10px',
    'backgroundColor': '#f0f8ff'
}, children=[
    # заголовок страницы
    html.H2("Исходные данные"),

    # кнопка назад
    html.A("Назад", href='/', style={
        'color': '#28a745',
        'textDecoration': 'none',
        'fontSize': '1.1em',
        'marginBottom': '100px'
    }),

    # таблица
    dash_table.DataTable(
        data=kdf.to_dict('records'),
        columns=[{"name": i, "id": i} for i in kdf.columns],
        style_cell={'textAlign': 'center', 'padding': '1px'},
        style_header={
            'backgroundColor': '#28a745',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_table={'width': '100%', 'margin': '0 auto', 'marginTop': '10px',}
    ),
])


# страница с визуализациями
dash_dashboard_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dashboard/',
    suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(style={
    'fontFamily': 'Sans-serif',
    'textAlign': 'center',
    'padding': '10px',
    'backgroundColor': '#f0f8ff'
}, children=[
    # заголовок страницы
    html.H2("📊 Аналитика данных"),

    # кнопка назад
    html.A("Назад", href='/', style={
        'color': '#28a745',
        'textDecoration': 'none',
        'fontSize': '1.1em',
        'marginBottom': '100px'
    }),

    # графики во вкладках
    dcc.Tabs([
        dcc.Tab(label="Самые частые вакансии", children=[
            dcc.Graph(figure=fig, style={'marginBottom': '0px', 'marginTop': '0px'}),  
        ]),
        dcc.Tab(label="Вакансии в области аналитики", children=[
            dcc.Graph(figure=fig2, style={'marginBottom': '0px', 'marginTop': '0px'}),  
        ])
    ], style={'marginTop': '10px', 'fontFamily': 'Sans-serif'}),

    # визуализация графа
    dcc.Graph(figure=fig3, style={'marginBottom': '10px', 'marginTop': '10px'}),])


# запуск приложения
if __name__ == '__main__':
    server.run(debug=False)
