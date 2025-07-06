import yfinance as yf
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings





# --- CONFIGURACIÓN INICIAL ---
# Ignorar advertencias comunes para una salida más limpia
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Configuración de estilo para los gráficos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)

# --- PASO 1: Definir el Universo de Activos para Argentina ---
tickers = [
    'GGAL', 'YPFD', 'PAMP', 'LOMA', 'BMA', 'MELI', 'IRS', 'CRESY',
    '^MERV', '^GSPC', 'EWZ',
    'ZS=F', 'ZC=F', 'CL=F', 'GC=F',
    '^TNX'
]

# --- PASO 2: Descargar y Preparar los Datos ---
print("Descargando datos de mercado...")
data = yf.download(tickers, start="2018-01-01", end="2025-01-01", auto_adjust=True)['Close']
data.dropna(axis=1, thresh=len(data) * 0.8, inplace=True)
data.dropna(axis=0, inplace=True)

print("Datos descargados y limpios. Tickers utilizados finalmente:")
print(data.columns.tolist())

log_returns = np.log(data / data.shift(1)).dropna()

# --- GRÁFICO 1: RED MST ESTÁTICA (Período Completo) ---
def plot_static_mst(log_returns_df):
    """
    Calcula y grafica el MST para todo el período de datos.
    """
    print("\nGenerando gráfico de la red MST estática...")
    corr_matrix = log_returns_df.corr()
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    G = nx.from_pandas_adjacency(dist_matrix)
    mst = nx.minimum_spanning_tree(G)

    # Convertir distancias de vuelta a correlaciones para las etiquetas (más intuitivo)
    edge_labels = {}
    for u, v in mst.edges():
        corr = corr_matrix.loc[u, v]
        edge_labels[(u, v)] = f"{corr:.2f}"

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(mst, k=0.75, iterations=50) # 'k' ajusta la distancia entre nodos

    nx.draw_networkx_nodes(mst, pos, node_size=2000, node_color='skyblue', alpha=0.9)
    nx.draw_networkx_edges(mst, pos, width=1.5, alpha=0.8, edge_color='gray')
    nx.draw_networkx_labels(mst, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels, font_size=9, font_color='black')

    plt.title('Red MST Estática del Mercado Argentino y Activos Relacionados', fontsize=20)
    plt.box(False)
    plt.show()

plot_static_mst(log_returns)


# --- PASO 3: Función para Extraer Métricas de Red ---
def extract_network_metrics(log_returns, window_size=60, step=5):
    metrics = []
    print(f"\nExtrayendo métricas de red con ventanas de {window_size} días...")
    for start in range(0, len(log_returns) - window_size, step):
        end = start + window_size
        window_data = log_returns.iloc[start:end]
        corr_matrix = window_data.corr()
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        G = nx.from_pandas_adjacency(dist_matrix)
        mst = nx.minimum_spanning_tree(G)
        metrics.append({
            "end_date": log_returns.index[end - 1],
            "avg_degree": np.mean(list(dict(mst.degree()).values())),
            "density": nx.density(mst),
            "avg_clustering": nx.average_clustering(mst),
            "avg_shortest_path": nx.average_shortest_path_length(mst)
        })
    print("Extracción de métricas completada.")
    return pd.DataFrame(metrics).set_index('end_date')

network_metrics_df = extract_network_metrics(log_returns)

# --- PASO 4: Definir la Variable Objetivo (Crisis) ---
print("\nDefiniendo la variable objetivo 'crisis'...")
if '^MERV' in log_returns.columns:
    future_window = 60
    future_returns = log_returns['^MERV'].rolling(window=future_window).sum().shift(-future_window)
    network_metrics_df['future_return'] = future_returns
    crisis_threshold = -0.10
    network_metrics_df['crisis'] = (network_metrics_df['future_return'] < crisis_threshold).astype(int)
    network_metrics_df.dropna(inplace=True)

    print(f"Definición de crisis: Retorno del MERVAL a {future_window} días < {crisis_threshold*100}%")
    print(f"Número de períodos de crisis identificados: {network_metrics_df['crisis'].sum()}")

    # --- PASO 5: Entrenar el Modelo de Machine Learning ---
    print("\nEntrenando el modelo RandomForest...")
    X = network_metrics_df[['avg_degree', 'density', 'avg_clustering', 'avg_shortest_path']]
    y = network_metrics_df['crisis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- PASO 6: Mostrar Resultados y Gráficos del Modelo ---
    print("\n--- Resultados del Modelo ---")
    print(classification_report(y_test, y_pred, target_names=['No Crisis (0)', 'Crisis (1)']))

    # --- GRÁFICO 2: EVOLUCIÓN DE MÉTRICAS Y CRISIS ---
    print("\nGenerando gráfico de evolución de métricas...")
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 12), sharex=True)
    fig.suptitle('Evolución de Métricas de Red en el Tiempo', fontsize=20)

    metrics_to_plot = ['avg_degree', 'density', 'avg_clustering', 'avg_shortest_path']
    crisis_dates = network_metrics_df[network_metrics_df['crisis'] == 1].index

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(network_metrics_df.index, network_metrics_df[metric], label=metric, color='royalblue')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        # Sombrear las áreas de crisis
        for date in crisis_dates:
            axes[i].axvspan(date - pd.Timedelta(days=future_window), date, color='red', alpha=0.2, lw=0)

    axes[0].legend(['Métrica', 'Período de Crisis Inminente'], loc='upper left')
    plt.xlabel('Fecha')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    # --- GRÁFICO 3: IMPORTANCIA DE LAS CARACTERÍSTICAS ---
    print("\nGenerando gráfico de importancia de características...")
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
    #plt.title('Importancia de las Métricas de Red para Predecir Crisis', fontsize=16)
    plt.xlabel('Importancia')
    plt.ylabel('Métrica (Característica)')
    plt.show()
