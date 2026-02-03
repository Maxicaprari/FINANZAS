import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.linalg import eigh
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42

variables_prohibidas = [
    'var_igual_periodo_anio_anterior', 'indice_serie',
    'indice_tendencia', 'serie_desestacionalizada',
    'variacion_interanual', 'var_interanual'
]

cols_to_drop = [col for col in df.columns if any(p in col.lower() for p in variables_prohibidas)]
df = df.drop(columns=cols_to_drop, errors='ignore')

if 'Fecha' in df_emae_1.columns:
    df_emae_1 = df_emae_1.copy()
    df_emae_1['Fecha'] = pd.to_datetime(df_emae_1['Fecha'], errors='coerce')
    df_emae_1 = df_emae_1.set_index('Fecha').sort_index()

fecha_corte = pd.Timestamp('2005-01-01')
df_emae_1_filtrado = df_emae_1[df_emae_1.index < fecha_corte].copy()

if 'Date' in df_emae.columns:
    df_emae = df_emae.copy()
    df_emae['Date'] = pd.to_datetime(df_emae['Date'], errors='coerce')
    df_emae = df_emae.set_index('Date').sort_index()

df_emae_filtrado = df_emae[df_emae.index >= fecha_corte].copy()

if 'emae_yoy' in df_emae_1_filtrado.columns:
    y_series_1 = df_emae_1_filtrado['emae_yoy'].dropna().copy()
else:
    y_series_1 = pd.Series(dtype=float, name='emae_yoy')

emae_col_name = 'Emae - sa_orig - current_prices_yoy'
if emae_col_name in df_emae_filtrado.columns:
    y_series_2 = df_emae_filtrado[emae_col_name].dropna().copy()
    y_series_2.name = 'emae_yoy'
else:
    possible_cols = [col for col in df_emae_filtrado.columns if 'emae' in col.lower() or 'yoy' in col.lower()]
    emae_col_name = possible_cols[0]
    y_series_2 = df_emae_filtrado[emae_col_name].dropna().copy()
    y_series_2.name = 'emae_yoy'

y_series = pd.concat([y_series_1, y_series_2]).sort_index()

if 'fecha' in df.columns:
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.set_index('fecha').sort_index()

y_series = y_series.dropna()
common_index = df.index.intersection(y_series.index)
df = df.loc[common_index]
y_series = y_series.loc[common_index]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())

rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(df_numeric, y_series)
importances = pd.Series(rf.feature_importances_, index=df_numeric.columns)
top_vars = importances.nlargest(20).index.tolist()
X_selected = df_numeric[top_vars].copy()

window_size = 12
T, p = X_selected.shape
Z_list, valid_indices = [], []

for t in range(window_size - 1, T):
    window = X_selected.iloc[t - window_size + 1:t + 1].values.flatten()
    Z_list.append(window)
    valid_indices.append(X_selected.index[t])

Z = np.array(Z_list)
Z_index = pd.DatetimeIndex(valid_indices)

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

max_components = min(Z_scaled.shape[0], Z_scaled.shape[1])
n_components_initial = min(max(5, 10), max_components)

pca = PCA(n_components=n_components_initial)
embeddings_full = pca.fit_transform(Z_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)

idx_high = np.where(cumvar >= 0.85)[0]
idx_low = np.where(cumvar >= 0.70)[0]

if len(idx_high) > 0:
    n_optimal = max(idx_high[0] + 1, 3)
elif len(idx_low) > 0:
    n_optimal = max(idx_low[0] + 1, 5)
else:
    n_optimal = min(5, max_components)

embeddings = embeddings_full[:, :n_optimal]

distances = pdist(embeddings, metric='euclidean')
distance_matrix = squareform(distances)
sparse_graph = csr_matrix(distance_matrix)
mst = minimum_spanning_tree(sparse_graph)
edge_lengths = mst.data

def spectral_clustering_temporal(embeddings, dates, distance_matrix):
    n_samples = len(embeddings)
    dates_numeric = np.array([(d - dates.min()).days for d in dates])
    sigma_spatial = np.median(pdist(embeddings)) * 0.4
    tau_temporal = np.median(pdist(dates_numeric.reshape(-1, 1), metric='cityblock')) * 0.25

    dist_spatial_matrix = cdist(embeddings, embeddings)
    sim_spatial_matrix = np.exp(-(dist_spatial_matrix**2) / (2 * sigma_spatial**2))
    dist_temporal_matrix = np.abs(dates_numeric[:, None] - dates_numeric[None, :])
    sim_temporal_matrix = np.exp(-dist_temporal_matrix / tau_temporal)
    affinity_matrix = sim_spatial_matrix * sim_temporal_matrix
    np.fill_diagonal(affinity_matrix, 1.0)

    spectral = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=RANDOM_STATE, n_init=20)
    labels = spectral.fit_predict(affinity_matrix)
    return labels, affinity_matrix

spectral_labels, spectral_affinity = spectral_clustering_temporal(embeddings, Z_index, distance_matrix)
regimes = spectral_labels
regime_series = pd.Series(regimes, index=Z_index)
y_aligned = y_series.reindex(Z_index)

k_regimes_ms = min(max(2, len(set(regimes))), 5)
mr_model = MarkovRegression(y_aligned.values, k_regimes=k_regimes_ms, trend="c", switching_variance=True)
mr_res = mr_model.fit(disp=False, maxiter=200)

regime_start_dates = {r: Z_index[regimes == r].min() for r in set(regimes)}
sorted_regimes = sorted(regime_start_dates.items(), key=lambda x: x[1])
regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
regimes = np.array([regime_mapping[r] for r in regimes])

regime_interpretations = {}
color_map = plt.cm.get_cmap('tab10')

for rid in sorted(set(regimes)):
    mask = regimes == rid
    regime_interpretations[rid] = {
        'color': color_map(rid % 10),
        'duration': mask.sum(),
        'mean_growth': y_aligned[mask].mean(),
        'std_growth': y_aligned[mask].std(),
        'min_growth': y_aligned[mask].min(),
        'max_growth': y_aligned[mask].max(),
        'dates': Z_index[mask]
    }

plt.rcParams.update({'font.size':12})
fig1, ax1 = plt.subplots(figsize=(20,9))
ax1.plot(y_aligned.index, y_aligned.values, linewidth=3)
ax1.axhline(0, linestyle='--')
for rid, info in regime_interpretations.items():
    ax1.axvspan(info['dates'].min(), info['dates'].max(), alpha=0.2, color=info['color'])
plt.tight_layout()
plt.savefig('regimenes_temporales.png', dpi=300)
plt.show()

fig2, axes2 = plt.subplots(2,2,figsize=(18,13))
labels = [f'Reg {i+1}' for i in regime_interpretations.keys()]
durations = [info['duration'] for info in regime_interpretations.values()]
means = [info['mean_growth'] for info in regime_interpretations.values()]
vols = [info['std_growth'] for info in regime_interpretations.values()]
colors = [info['color'] for info in regime_interpretations.values()]

axes2[0,0].barh(labels, durations, color=colors)
axes2[0,1].bar(labels, means, color=colors)
axes2[1,0].bar(labels, vols, color=colors)
for i,(rid,info) in enumerate(regime_interpretations.items()):
    axes2[1,1].plot([i,i],[info['min_growth'],info['max_growth']], color=info['color'], linewidth=6)
plt.tight_layout()
plt.savefig('estadisticos_regimenes.png', dpi=300)
plt.show()

fig3, ax3 = plt.subplots(figsize=(18,18))
if embeddings.shape[1] > 2:
    emb2d = PCA(n_components=2).fit_transform(embeddings)
else:
    emb2d = embeddings[:, :2]

mst_graph = nx.Graph()
rows, cols = mst.nonzero()
for i,j in zip(rows,cols):
    if i<j:
        mst_graph.add_edge(i,j)

pos = {i:(emb2d[i,0],emb2d[i,1]) for i in range(len(emb2d))}
nx.draw_networkx_edges(mst_graph,pos,alpha=0.4,ax=ax3)
for rid,info in regime_interpretations.items():
    nodes = np.where(regimes==rid)[0]
    nx.draw_networkx_nodes(mst_graph,pos,nodelist=nodes,node_color=[info['color']],ax=ax3)
ax3.axis('off')
plt.tight_layout()
plt.savefig('mst_network.png', dpi=300)
plt.show()
