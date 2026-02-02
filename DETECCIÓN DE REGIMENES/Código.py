import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.linalg import eigh
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib.pyplot as plt
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

variables_prohibidas = [
    'var_igual_periodo_anio_anterior', 'indice_serie',
    'indice_tendencia', 'serie_desestacionalizada',
    'variacion_interanual', 'var_interanual'
]

cols_to_drop = [col for col in df.columns
                if any(p in col.lower() for p in variables_prohibidas)]

if cols_to_drop:
    df = df.drop(columns=cols_to_drop, errors='ignore')

if 'Fecha' in df_emae.columns:
    df_emae = df_emae.copy()
    df_emae['Fecha'] = pd.to_datetime(df_emae['Fecha'], errors='coerce')
    df_emae = df_emae.set_index('Fecha').sort_index()

if 'fecha' in df.columns:
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.set_index('fecha').sort_index()

emae_nivel = df_emae['emae_empalmado'].dropna().copy()
y_series = emae_nivel.pct_change(12) * 100
y_series = y_series.dropna()

common_index = df.index.intersection(y_series.index)
df = df.loc[common_index]
y_series = y_series.loc[common_index]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols].copy()
df_numeric = df_numeric.fillna(df_numeric.median())

rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(df_numeric, y_series)
importances = pd.Series(rf.feature_importances_, index=df_numeric.columns)
top_vars = importances.nlargest(20).index.tolist()

X_selected = df_numeric[top_vars].copy()

window_size = 12
T, p = X_selected.shape
Z_list = []
valid_indices = []

for t in range(window_size - 1, T):
    window = X_selected.iloc[t - window_size + 1:t + 1].values.flatten()
    Z_list.append(window)
    valid_indices.append(X_selected.index[t])

Z = np.array(Z_list)
Z_index = pd.DatetimeIndex(valid_indices)

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

max_components = min(Z_scaled.shape[0], Z_scaled.shape[1])
MIN_COMPONENTS = 5
n_components_initial = min(max(MIN_COMPONENTS, 10), max_components)

pca = PCA(n_components=n_components_initial)
embeddings_full = pca.fit_transform(Z_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
target_var_high = 0.85
target_var_low = 0.70

idx_high = np.where(cumvar >= target_var_high)[0]
idx_low = np.where(cumvar >= target_var_low)[0]

if len(idx_high) > 0:
    n_optimal = max(idx_high[0] + 1, 3)
elif len(idx_low) > 0:
    n_optimal = max(idx_low[0] + 1, MIN_COMPONENTS)
else:
    n_optimal = min(MIN_COMPONENTS, max_components)

embeddings = embeddings_full[:, :n_optimal]

distances = pdist(embeddings, metric='euclidean')
distance_matrix = squareform(distances)

sparse_graph = csr_matrix(distance_matrix)
mst = minimum_spanning_tree(sparse_graph)
edge_lengths = mst.data
edge_lengths_sorted = np.sort(edge_lengths)[::-1]

def spectral_clustering_temporal(embeddings, dates, n_clusters=None,
                                 sigma_spatial=None, tau_temporal=None,
                                 auto_tune=True):

    n_samples = len(embeddings)
    dates_numeric = np.array([(d - dates.min()).days for d in dates])

    if sigma_spatial is None or auto_tune:
        distances_spatial = pdist(embeddings, metric='euclidean')
        sigma_spatial = np.median(distances_spatial) * 0.5

    if tau_temporal is None or auto_tune:
        distances_temporal = pdist(dates_numeric.reshape(-1, 1), metric='cityblock')
        tau_temporal = np.median(distances_temporal) * 0.3

    dist_spatial_matrix = cdist(embeddings, embeddings, metric='euclidean')
    sim_spatial_matrix = np.exp(-(dist_spatial_matrix ** 2) / (2 * sigma_spatial ** 2))

    dist_temporal_matrix = np.abs(dates_numeric[:, None] - dates_numeric[None, :])
    sim_temporal_matrix = np.exp(-dist_temporal_matrix / tau_temporal)

    affinity_matrix = sim_spatial_matrix * sim_temporal_matrix
    np.fill_diagonal(affinity_matrix, 1.0)

    if n_clusters is None or auto_tune:
        laplacian = np.diag(affinity_matrix.sum(axis=1)) - affinity_matrix
        n_eig = min(20, affinity_matrix.shape[0] - 1)
        eigenvalues, _ = eigh(laplacian, subset_by_index=[0, n_eig])
        eigenvalues = np.sort(eigenvalues)
        gaps = np.diff(eigenvalues)
        significant_gaps = gaps[gaps > np.percentile(gaps, 50)]
        if len(significant_gaps) > 0:
            max_gap_idx = np.where(gaps == significant_gaps.max())[0][0]
            n_clusters = max_gap_idx + 2
        else:
            n_clusters = 4
        n_clusters = max(3, min(n_clusters, 8))

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = spectral.fit_predict(affinity_matrix)

    params = {
        'sigma_spatial': sigma_spatial,
        'tau_temporal': tau_temporal,
        'n_clusters': n_clusters
    }

    return labels, affinity_matrix, params

spectral_labels, spectral_affinity, spectral_params = spectral_clustering_temporal(
    embeddings, Z_index, n_clusters=None, auto_tune=True
)

regimes = spectral_labels
use_spectral = True

n_regimes = len(set(regimes))

regime_series = pd.Series(regimes, index=Z_index, name='regime')
y_aligned = y_series.reindex(Z_index)

VALIDATE_MARKOV = True

if VALIDATE_MARKOV:
    try:
        k_regimes_ms = min(max(2, n_regimes), 5)
        mr_model = MarkovRegression(
            y_aligned.values,
            k_regimes=k_regimes_ms,
            trend="c",
            switching_variance=True
        )
        mr_res = mr_model.fit(disp=False, maxiter=200)

        smoothed_probs = np.vstack([
            mr_res.smoothed_marginal_probabilities[i]
            for i in range(k_regimes_ms)
        ]).T
        markov_regimes = smoothed_probs.argmax(axis=1)
    except Exception:
        markov_regimes = None

regime_interpretations = {}
color_map = plt.cm.get_cmap('tab10')

regime_start_dates = {}
for regime_id in set(regimes):
    regime_mask = regimes == regime_id
    regime_dates = Z_index[regime_mask]
    regime_start_dates[regime_id] = regime_dates.min()

sorted_regimes = sorted(regime_start_dates.items(), key=lambda x: x[1])
regime_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_regimes)}

regimes_renumbered = np.array([regime_mapping[r] for r in regimes])
regimes = regimes_renumbered

for new_regime_id in sorted(set(regimes)):
    regime_mask = regimes == new_regime_id
    regime_dates = Z_index[regime_mask]
    regime_y = y_aligned[regime_mask]

    duration = regime_mask.sum()
    mean_growth = regime_y.mean()
    std_growth = regime_y.std()
    min_growth = regime_y.min()
    max_growth = regime_y.max()

    color = color_map(new_regime_id % 10)

    regime_interpretations[new_regime_id] = {
        'label': f"Régimen {new_regime_id + 1}",
        'color': color,
        'duration': duration,
        'mean_growth': mean_growth,
        'std_growth': std_growth,
        'min_growth': min_growth,
        'max_growth': max_growth,
        'period_start': regime_dates.min(),
        'period_end': regime_dates.max(),
        'dates': regime_dates
    }
