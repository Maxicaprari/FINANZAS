
tickers_existentes = [
    "GGAL","BMA","YPF","PAM","TGS","TEO","IRS","CRESY","MELI",
    "^MERV","EWZ","^GSPC","GC=F","CL=F","ZS=F","ZC=F","^TNX"
]


tickers_nuevos = [
    "BBAR","SUPV","CEPU","LOMA","DESP",
    "ARGT","ILF","EEM",
    "ARS=X","USDBRL=X","ZW=F"
]


TICKERS = tickers_existentes + tickers_nuevos

SOURCE = "yfinance"      # "yfinance" o "api"
START = "2010-01-01"
END   = "2025-01-01"



# API (si usás SOURCE="api"): debe devolver columnas ['fecha','ticker','precio_cierre']
API_ENDPOINT = "https://TU_API/precios"   # <-- reemplazar
API_HEADERS = {"Authorization": "Bearer TU_TOKEN"}  # opcional
API_PARAMS = {"fecha_desde": START, "fecha_hasta": END}  # opcional

# DEFINIMOS LAS VENTANAS
WINDOW_SIZE = 60     
STEP = 5              
CRISIS_FUTURE_WINDOW = 60  
CRISIS_THRESHOLD = -0.10    


N_CLUSTERS = 3
RANDOM_STATE = 42


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if SOURCE == "yfinance":
    import yfinance as yf


#Descargamos los datos

def fetch_from_yfinance(tickers, start, end):
    print("Descargando datos desde yfinance...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    print(f"Descargados {data.shape[1]} tickers y {data.shape[0]} filas.")
    return data

def fetch_from_api(endpoint, headers=None, params=None):

    import requests
    print("Descargando datos desde API...")
    r = requests.get(endpoint, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    # Intento 1: JSON
    try:
        raw = pd.DataFrame(r.json())
    except ValueError:
        from io import StringIO
        raw = pd.read_csv(StringIO(r.text))
    cols = {c.lower(): c for c in raw.columns}
    ren = {}
    for c in raw.columns:
        cl = c.lower()
        if cl in ["fecha", "date"]:
            ren[c] = "fecha"
        elif cl in ["ticker", "symbol"]:
            ren[c] = "ticker"
        elif cl in ["precio_cierre", "close", "precio"]:
            ren[c] = "precio_cierre"
    raw = raw.rename(columns=ren)
    # Pivot a matriz de precios (filas=fecha, cols=ticker)
    raw['fecha'] = pd.to_datetime(raw['fecha'])
    data = raw.pivot(index="fecha", columns="ticker", values="precio_cierre").sort_index()
    print(f"API -> matriz: {data.shape[0]} filas x {data.shape[1]} columnas.")
    return data

if SOURCE == "yfinance":
    prices = fetch_from_yfinance(TICKERS, START, END)
else:
    prices = fetch_from_api(API_ENDPOINT, headers=API_HEADERS, params=API_PARAMS)


prices = prices.sort_index()
prices = prices.loc[~prices.index.duplicated(keep="first")]
prices = prices.dropna(axis=1, thresh=len(prices)*0.80)   # al menos 80% de datos por columna
prices = prices.dropna(axis=0)                            # drop filas con NaN restantes

print("Tickers finales:", list(prices.columns))


#Retornos, métricas base y ventanas móviles

log_returns = np.log(prices / prices.shift(1)).dropna()

def rolling_metrics(returns, window=60, step=5):
    rows = []
    idx = returns.index
    for start in range(0, len(idx) - window, step):
        end = start + window
        win = returns.iloc[start:end]
        end_date = idx[end-1]
        mu = win.mean() * 252
        vol = win.std() * np.sqrt(252)
        sharpe = mu / (vol.replace(0, np.nan))
        ret_cum = win.sum()  # aprox del log-retorno acumulado en la ventana
        tmp = pd.DataFrame({
            "end_date": end_date,
            "ticker": win.columns,
            "ret_log_cum_window": ret_cum.values,
            "mu_ann": mu.values,
            "vol_ann": vol.values,
            "sharpe": sharpe.values
        })
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out.set_index(["end_date", "ticker"], inplace=True)
    return out


def extract_network_metrics_enhanced(log_returns, window_size=60, step=5):
    metrics = []
    idx = log_returns.index
    print(f"\nExtrayendo métricas de red con ventanas de {window_size}d y step {step}d...")
    for start in range(0, len(idx)-window_size, step):
        end = start + window_size
        end_date = idx[end-1]
        w = log_returns.iloc[start:end]
        corr = w.corr()
        dist = np.sqrt(2*(1-corr)).fillna(0.0)

        # Grafo completo
        G = nx.from_pandas_adjacency(dist)
        avg_edge_weight_full = (G.size(weight='weight') / G.number_of_edges()) if G.number_of_edges() > 0 else 0.0
        avg_clustering_weighted = nx.average_clustering(G, weight='weight')

        # MST
        mst = nx.minimum_spanning_tree(G)
        mst_total_weight = mst.size(weight='weight')
        # Nota: average_shortest_path_length requiere grafo conectado.
        try:
            mst_avg_shortest_path = nx.average_shortest_path_length(mst, weight='weight')
        except nx.NetworkXError:
            mst_avg_shortest_path = np.nan

        # Agrego algo de "nivel de mercado": promedio de correlaciones off-diagonal
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg_corr_offdiag = corr.where(mask).stack().mean()

        metrics.append({
            "end_date": end_date,
            "avg_edge_weight_full": avg_edge_weight_full,
            "avg_clustering_weighted": avg_clustering_weighted,
            "mst_total_weight": mst_total_weight,
            "mst_avg_shortest_path": mst_avg_shortest_path,
            "avg_corr_offdiag": avg_corr_offdiag
        })
    df = pd.DataFrame(metrics).set_index("end_date")
    print("Listo.")
    return df

network_metrics_df = extract_network_metrics_enhanced(
    log_returns, window_size=WINDOW_SIZE, step=STEP
)

# PCA + KMeans 
def pca_kmeans_on_assets(returns_df, n_clusters=3, random_state=42):
    """
    PCA sobre matriz activos x features (usamos correlaciones/promedios).
    Lo más directo: PCA sobre matriz de returns transpuesta.
    """
    # Estandarización simple: como son log-returns diarios, PCA en transpuesta
    X = returns_df.T.values
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)
    pca_df = pd.DataFrame(Z, index=returns_df.columns, columns=["PC1", "PC2"])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    pca_df["Cluster"] = km.fit_predict(pca_df[["PC1","PC2"]])

    plt.figure(figsize=(10,7))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, s=200, palette="Set2")
    for t in pca_df.index:
        plt.text(pca_df.loc[t,"PC1"]+0.01, pca_df.loc[t,"PC2"]+0.01, t, fontsize=9)
    plt.title("Clusters de acciones (PCA + KMeans)")
    plt.legend(title="Cluster")
    plt.show()
    return pca_df, pca

pca_clusters_df, pca_model = pca_kmeans_on_assets(log_returns, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)




#Dendrograma jerárquico

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    corr = log_returns.corr()
    # Distancia tipo "1-corr"
    dist = 1 - corr
    # Para linkage usamos vector condensado:
    from scipy.spatial.distance import squareform
    dist_vec = squareform(dist.values, checks=False)
    Z = linkage(dist_vec, method="ward")
    plt.figure(figsize=(12,6))
    dendrogram(Z, labels=corr.columns, leaf_rotation=90)
    plt.title("Dendrograma jerárquico (1 - correlación)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("No se pudo generar dendrograma (quizás falta scipy):", e)





print("\n✅ Flujo completo terminado.")
print("Dimensiones:")
print(" - prices:", prices.shape)
print(" - log_returns:", log_returns.shape)
print(" - pca_clusters_df:", pca_clusters_df.shape)


# GRAFICOS ##################################################################



# PCA: Scree Plot, Biplot, Heatmap y Clusters

from sklearn.preprocessing import StandardScaler


X_scaled = StandardScaler().fit_transform(log_returns.values)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

#Scree Plot 
plt.figure(figsize=(8,6))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Explicada Acumulada (%)")
plt.title("Scree Plot (Varianza Explicada)")
plt.grid(True)
plt.show()

#Biplot
def biplot(scores, coeffs, labels=None):
    xs, ys = scores[:,0], scores[:,1]
    plt.figure(figsize=(9,7))
    plt.scatter(xs, ys, alpha=0.6)
    for i in range(coeffs.shape[0]):
        plt.arrow(0, 0, coeffs[i,0]*max(xs), coeffs[i,1]*max(ys),
                  color='r', alpha=0.7, head_width=0.05)
        if labels is None:
            plt.text(coeffs[i,0]*max(xs)*1.1, coeffs[i,1]*max(ys)*1.1, f"Var{i+1}", color='r')
        else:
            plt.text(coeffs[i,0]*max(xs)*1.1, coeffs[i,1]*max(ys)*1.1, labels[i], color='r')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot (PC1 vs PC2)")
    plt.grid(True)
    plt.show()

biplot(X_pca, pca.components_.T, labels=log_returns.columns)

#Heatmap de correlaciones
plt.figure(figsize=(10,8))
sns.heatmap(log_returns.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación entre retornos")
plt.show()

#Proyección temporal en espacio PCA
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=range(len(log_returns)), cmap="viridis", alpha=0.7)
plt.colorbar(label="Índice temporal")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proyección temporal en componentes principales")
plt.show()

#Loading Matrix 
loading_matrix = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(log_returns.columns))],
    index=log_returns.columns
)

print("Importancia de las variables en los primeros componentes:")
display(loading_matrix.iloc[:,:2].sort_values("PC1", ascending=False))

#Clusterización KMeans en espacio PCA
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit(X_scaled)
labels = kmeans.labels_

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="Set1", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusterización de acciones (KMeans en espacio PCA)")
plt.show()

###### GRAFICOS MEJORADOS 

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(12,10))


kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit(X_pca[:, :2])
labels = kmeans.labels_


dist = np.sqrt(X_pca[:,0]**2 + X_pca[:,1]**2)
top_idx = dist.argsort()[-15:]  # top 15 observaciones más alejadas


plt.plot(X_pca[:,0], X_pca[:,1], color='lightgray', alpha=0.5, linewidth=0.7)


scatter = plt.scatter(
    X_pca[:,0], X_pca[:,1],
    c=labels, cmap="viridis", alpha=0.7, s=40
)


for idx in top_idx:
    plt.annotate(log_returns.index[idx].strftime('%Y-%m-%d'),
                 (X_pca[idx,0], X_pca[idx,1]),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center', fontsize=8, color='darkslategray')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1],
            c="black", marker="X", s=200, label="Centroides")

plt.legend()


plt.colorbar(scatter, label="Cluster")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proyección temporal en componentes principales (paleta suave)")

plt.show()




sns.set_style("whitegrid")
sns.set_context("talk")

# Scree Plot 
def scree_plot(pca):
    plt.figure(figsize=(10, 6))
    var_ratio = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratio)

    bars = plt.bar(range(1, len(var_ratio)+1), var_ratio,
                   alpha=0.7, color=sns.color_palette("Spectral", len(var_ratio)),
                   label="Varianza individual")
    plt.plot(range(1, len(var_ratio)+1), cum_var, marker='o', color="black", lw=2,
             label="Varianza acumulada")

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{var_ratio[i]:.1f}%", ha='center', fontsize=9, color="dimgray")

    plt.xlabel("Número de Componentes", fontsize=12)
    plt.ylabel("Varianza Explicada (%)", fontsize=12)
    plt.title("Scree Plot (Varianza Explicada)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Biplot Mejorado
def biplot(scores, coeffs, labels=None):
    xs, ys = scores[:, 0], scores[:, 1]
    plt.figure(figsize=(12, 9))
    plt.scatter(xs, ys, alpha=0.7, s=50, c=np.arange(len(xs)), cmap="viridis")

    for i in range(coeffs.shape[0]):
        plt.arrow(0, 0, coeffs[i, 0]*max(xs), coeffs[i, 1]*max(ys),
                  color='darkslateblue', alpha=0.7, head_width=0.15, linewidth=2)
        name = labels[i] if labels is not None else f"Var{i+1}"
        plt.text(coeffs[i, 0]*max(xs)*1.15,
                 coeffs[i, 1]*max(ys)*1.15,
                 name, color='darkslateblue', fontsize=10)

    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.title("Biplot (PC1 vs PC2)", fontsize=14, fontweight="bold")
    plt.colorbar(label="Índice temporal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


#  Heatmap de correlaciones estilizado 
def plot_corr_heatmap(df):
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="crest",
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    plt.title("Matriz de correlación entre retornos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Helper para etiquetar fechas desde el índice
def _idx_label(idx_value):
    try:
        return pd.to_datetime(idx_value).strftime('%Y-%m-%d')
    except Exception:
        return str(idx_value)

#  Proyección temporal en espacio PCA (usa df.index) 
def temporal_projection(X_pca, df, n_anotaciones=8):
    idx = df.index  # usa el índice del DataFrame como fechas
    plt.figure(figsize=(11, 9))

    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=np.arange(len(idx)), cmap="plasma", alpha=0.8, s=60,
                     edgecolor="k", linewidth=0.3)
    plt.plot(X_pca[:, 0], X_pca[:, 1], color="lightgray", linewidth=0.7, alpha=0.5)

    plt.colorbar(sc, label="Índice temporal")
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.title("Proyección temporal en componentes principales", fontsize=14, fontweight="bold")

    # Etiquetar outliers por distancia al origen en el plano PC1-PC2
    if len(idx) > 0 and n_anotaciones > 0:
        dist = np.sqrt(X_pca[:, 0]**2 + X_pca[:, 1]**2)
        top_idx = dist.argsort()[-min(n_anotaciones, len(dist)):]
        for i in top_idx:
            plt.text(X_pca[i, 0], X_pca[i, 1], _idx_label(idx[i]),
                     fontsize=8, color="dimgray")

    plt.tight_layout()
    plt.show()


#  Clusterización con centroides + trayectorias 
def cluster_projection(X_pca, labels, df, kmeans=None, n_anotaciones=10):
    idx = df.index  # usa el índice del DataFrame como fechas
    plt.figure(figsize=(11, 9))

    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=labels, cmap="Set2", alpha=0.85, s=60,
                     edgecolor="k", linewidth=0.3)
    plt.plot(X_pca[:, 0], X_pca[:, 1], color="gray", linewidth=0.6, alpha=0.4)

    # Centroides
    if kmeans is not None:
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    marker="X", s=320, c="black", label="Centroides", edgecolor="white")

    # Anotar outliers por distancia 
    if len(idx) > 0 and n_anotaciones > 0:
        dist = np.sqrt(X_pca[:, 0]**2 + X_pca[:, 1]**2)
        top_idx = dist.argsort()[-min(n_anotaciones, len(dist)):]
        for i in top_idx:
            plt.text(X_pca[i, 0], X_pca[i, 1], _idx_label(idx[i]),
                     fontsize=8, color="darkslategray")

    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.title("Clusterización de acciones en espacio PCA", fontsize=14, fontweight="bold")
    if kmeans is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()


scree_plot(pca)
biplot(X_pca, pca.components_.T, labels=log_returns.columns)
plot_corr_heatmap(log_returns)
temporal_projection(X_pca, log_returns)                 
cluster_projection(X_pca, labels, log_returns, kmeans)  



