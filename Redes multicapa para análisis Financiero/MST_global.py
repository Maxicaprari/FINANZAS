data = yf.download(tickers, start="2020-01-01", end="2023-01-01", group_by='ticker', auto_adjust=True)
# Construir DataFrame limpio de precios de cierre ajustados
close_data = pd.DataFrame()
for ticker in tickers:
    if ticker in data.columns.levels[0]:  # chequea que el ticker esté en los datos
        close_data[ticker] = data[ticker]['Close']

#print(close_data.head())


# Calcular retornos logarítmicos
log_returns = np.log(close_data / close_data.shift(1)).dropna()

# Calcular matriz de correlación
corr_matrix = log_returns.corr()

# Convertir correlación a distancia
dist_matrix = np.sqrt(2 * (1 - corr_matrix))

# Crear grafo completo
G = nx.Graph()
for i in dist_matrix.columns:
    for j in dist_matrix.columns:
        if i != j:
            G.add_edge(i, j, weight=dist_matrix.loc[i, j])

# Crear Minimum Spanning Tree (MST)
mst = nx.minimum_spanning_tree(G)

# Dibujar MST
pos = nx.spring_layout(mst, seed=42)
plt.figure(figsize=(12, 8))
nx.draw(mst, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
edge_labels = nx.get_edge_attributes(mst, 'weight')
nx.draw_networkx_edge_labels(mst, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title("Red MST de correlaciones entre índices globales (2020–2023)")
plt.grid(True)
plt.show()






