import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


data = yf.download(tickers, start="2020-01-01", end="2023-01-01", group_by='ticker', auto_adjust=True)
# Construir DataFrame limpio de precios de cierre ajustados
close_data = pd.DataFrame()
for ticker in tickers:
    if ticker in data.columns.levels[0]:  # chequea que el ticker esté en los datos
        close_data[ticker] = data[ticker]['Close']

print(close_data.head())


# Calcular retornos logarítmicos
log_returns = np.log(close_data / close_data.shift(1)).dropna()

# Calcular matriz de correlación
corr_matrix = log_returns.corr()



import matplotlib.pyplot as plt

# Definir los dos períodos
period1 = log_returns.loc["2020-03-01":"2021-03-01"]
period2 = log_returns.loc["2022-01-01":"2022-12-31"]

def build_mst(log_ret_period):
    corr_matrix = log_ret_period.corr()
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    G = nx.Graph()
    for i in dist_matrix.columns:
        for j in dist_matrix.columns:
            if i != j:
                G.add_edge(i, j, weight=dist_matrix.loc[i, j])
    mst = nx.minimum_spanning_tree(G)
    return mst, corr_matrix


def plot_mst(mst, title):
    pos = nx.spring_layout(mst, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(mst, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray')
    nx.draw_networkx_edge_labels(mst, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in mst.edges(data=True)}, font_size=7)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Construcción y visualización
mst1, corr1 = build_mst(period1)
mst2, corr2 = build_mst(period2)

plot_mst(mst1, "MST - Período 1: Pandemia (2020-2021)")
plot_mst(mst2, "MST - Período 2: Suba de tasas (2022)")



def network_metrics(mst, corr_matrix):
    return {
        "avg_degree": np.mean([deg for _, deg in dict(mst.degree()).items()]),
        "density": nx.density(mst),
        "avg_clustering": nx.average_clustering(mst),
        "avg_shortest_path": nx.average_shortest_path_length(mst),
        "avg_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean(),
        "diameter": nx.diameter(mst)
    }

metrics1 = network_metrics(mst1, corr1)
metrics2 = network_metrics(mst2, corr2)

comparison_df = pd.DataFrame([metrics1, metrics2], index=["2020-2021", "2022"])
print(comparison_df)
