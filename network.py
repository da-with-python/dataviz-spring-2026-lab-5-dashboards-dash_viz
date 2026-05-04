import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain
import math
from pprint import pprint


def get_keywords(df, n_keywords=5):
    """
    Принимает на вход датафрейм с вакансиями и полем обработанных навыков.
    Возвращает датафрейм, состоящий из двух столбцов: название вакансии и столбец с ключевыми словами.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["tokens"].tolist())
    Xdf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
    
    def get_highest_tfidf(data, words, n=5):
        data = list(zip(words, data))
        return [x[0] for x in sorted(data, key=lambda x: x[1], reverse=True)[:n]]
    
    Xdf["keywords"] = Xdf.apply(lambda x: get_highest_tfidf(x, vectorizer.get_feature_names_out(), n=n_keywords), axis=1)
    kdf = pd.concat([df[["title"]].reset_index(), Xdf[["keywords"]]], axis=1)[["title", "keywords"]]
    # put your code here
    return kdf


def create_network(df):
    """
    Принимает на вход датафрейм с вакансиями и ключевыми словами.
    Возвращает список кортежей из пар вакансий и количества их общих ключевых слов.
    Вид кортежа внутри списка ожидается такой: (ребро1, ребро2, {'weight': вес_ребра})
    """

    vac_edges = []
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            if df.iloc[i, 0] != df.iloc[j, 0] and not set(df.iloc[i, 1]).isdisjoint(df.iloc[j, 1]):
                vac_edges.append(((df.iloc[i, 0], df.iloc[j, 0]), len(set(df.iloc[i, 1]).intersection(df.iloc[j, 1]))))
    vac_edges = [(tuple(sorted(edge[0])), edge[1]) for edge in vac_edges]
    weighted_edges = [(edge[0][0], edge[0][1], {"weight": edge[1]}) for edge in set(vac_edges)]
    return weighted_edges


def plot_network(vac_edges):
    """
    Строит визуализацию графа с помощью matplotlib.
    """
    G = nx.Graph()
    G.add_edges_from(vac_edges)
    nx.draw(G, with_labels=False, font_weight='bold', node_size=30)
    plt.show()


def get_communities(graph):
    """
    Извлекает сообщества из графа и фильтрует их так, 
    чтобы оставались только сообщества с более чем 5 узлами.
    Возвращает граф и сообщества в формате датафрейма.
    """
    communities = nx.community.louvain_communities(graph, resolution=1.2)
    comm_data = [{"n_of_nodes": len(comm), "nodes": comm} for comm in communities]
    cdf = pd.DataFrame(comm_data)
    nodes = list(chain(*cdf["nodes"].tolist()))
    S = graph.subgraph(nodes)
    communities = cdf.query("n_of_nodes>20")["nodes"].tolist()
    nodes = list(chain(*cdf.query("n_of_nodes>5")["nodes"].tolist()))
    s = S.subgraph(nodes)
    return communities, s


def create_community_node_colors(graph, communities):
    colors = list(set(mcolors.TABLEAU_COLORS.values()))
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


def plot_communities(communities, graph):
    """
    Строит интерактивную визуализацию графа с сообществами с помощью plotly.
    """
    pos = nx.spring_layout(graph, iterations=1000, seed=30, k=3/math.sqrt(len(graph)), scale=10.0)

    # edges coordinates
    x_nodes, y_nodes = zip(*pos.values())
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # degree labels
    node_labels = [f"Node {n}<br>Degree: {graph.degree(n)}" for n in graph.nodes()]
    node_degrees = [graph.degree(n) for n in graph.nodes()]
    node_colors_list = create_community_node_colors(graph, communities)

    fig = go.Figure()
    # add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="gray"),
    ))

    # add nodes
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers",
        marker=dict(size=[deg*0.5 for deg in node_degrees], color=node_colors_list),
        hoverinfo="text",
        text=node_labels
    ))

    # background settings
    fig.update_layout(
        title=f"Самые крупные кластеры вакансий",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    # fig.show()
    return fig
