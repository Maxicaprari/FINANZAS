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
from collections import Counter
import warnings

warnings.filterwarnings('ignore')





# PREPROCESAMIENTO


print("\n" + "=" * 70)
print("1. PREPROCESAMIENTO DE DATOS")
print("=" * 70)

# Limpieza
variables_prohibidas = [
    'var_igual_periodo_anio_anterior', 'indice_serie',
    'indice_tendencia', 'serie_desestacionalizada',
    'variacion_interanual', 'var_interanual'
]

cols_to_drop = [col for col in df.columns
                if any(p in col.lower() for p in variables_prohibidas)]

if cols_to_drop:
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"✅ Variables eliminadas: {len(cols_to_drop)}")

# Alineación temporal
if 'Fecha' in df_emae.columns:
    df_emae = df_emae.copy()
    df_emae['Fecha'] = pd.to_datetime(df_emae['Fecha'], errors='coerce')
    df_emae = df_emae.set_index('Fecha').sort_index()

if 'fecha' in df.columns:
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.set_index('fecha').sort_index()

# Variable objetivo
emae_nivel = df_emae['emae_empalmado'].dropna().copy()
y_series = emae_nivel.pct_change(12) * 100
y_series = y_series.dropna()

common_index = df.index.intersection(y_series.index)
df = df.loc[common_index]
y_series = y_series.loc[common_index]

print(f"✅ Observaciones: {len(common_index)}")
print(f"✅ Período: {y_series.index.min().strftime('%Y-%m')} → {y_series.index.max().strftime('%Y-%m')}")

# Seleccionar variables numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols].copy()
df_numeric = df_numeric.fillna(df_numeric.median())

# Seleccionar top 20 variables más importantes
rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(df_numeric, y_series)
importances = pd.Series(rf.feature_importances_, index=df_numeric.columns)
top_vars = importances.nlargest(20).index.tolist()

X_selected = df_numeric[top_vars].copy()
print(f"✅ Variables seleccionadas: {len(top_vars)}")
print(f"   Top 5: {', '.join(top_vars[:5])}")


# CONSTRUCCIÓN DE VENTANAS TEMPORALES


print("\n" + "=" * 70)
print("2. CONSTRUCCIÓN DE EMBEDDINGS TEMPORALES")
print("=" * 70)

window_size = 12  # 12 meses

T, p = X_selected.shape
Z_list = []
valid_indices = []

for t in range(window_size - 1, T):
    window = X_selected.iloc[t - window_size + 1:t + 1].values.flatten()
    Z_list.append(window)
    valid_indices.append(X_selected.index[t])

Z = np.array(Z_list)
Z_index = pd.DatetimeIndex(valid_indices)

print(f"✅ Ventanas temporales: {Z.shape[0]}")
print(f"✅ Dimensión por ventana: {Z.shape[1]}")


# PCA MEJORADO - MÍNIMO 5 COMPONENTES


print("\n" + "=" * 70)
print("2.5. REDUCCIÓN DIMENSIONAL CON PCA (MEJORADO)")
print("=" * 70)

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

# Calcular número máximo de componentes posibles
max_components = min(Z_scaled.shape[0], Z_scaled.shape[1])
print(f"   Máximo de componentes posibles: {max_components}")

# Forzar MÍNIMO 5 componentes (o el máximo disponible si es menor)
MIN_COMPONENTS = 5
n_components_initial = min(max(MIN_COMPONENTS, 10), max_components)

pca = PCA(n_components=n_components_initial)
embeddings_full = pca.fit_transform(Z_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"   Varianza explicada por {n_components_initial} componentes: {cumvar[-1]:.2%}")

target_var_high = 0.85
target_var_low = 0.70

idx_high = np.where(cumvar >= target_var_high)[0]
idx_low = np.where(cumvar >= target_var_low)[0]

if len(idx_high) > 0:
    n_optimal = max(idx_high[0] + 1, 3)
    print(f"   ✓ Estrategia: Alcanza {target_var_high:.0%} varianza")
elif len(idx_low) > 0:
    n_optimal = max(idx_low[0] + 1, MIN_COMPONENTS)
    print(f"   ✓ Estrategia: Alcanza {target_var_low:.0%} varianza, forzado a mínimo {MIN_COMPONENTS}")
else:
    n_optimal = min(MIN_COMPONENTS, max_components)
    print(f"   ⚠ Estrategia: Varianza baja, usando {n_optimal} componentes (mínimo requerido)")

embeddings = embeddings_full[:, :n_optimal]

print(f"\n✅ Componentes principales seleccionados: {n_optimal}")
print(f"✅ Varianza explicada: {cumvar[n_optimal - 1]:.2%}")
print(f"   Dimensión embeddings: {embeddings.shape}")

print(f"\n   Varianza por componente:")
for i in range(min(n_optimal, 10)):
    var_pct = pca.explained_variance_ratio_[i] * 100
    cumvar_pct = cumvar[i] * 100
    print(f"      PC{i + 1}: {var_pct:6.2f}% (acum: {cumvar_pct:6.2f}%)")


#CONSTRUCCIÓN DEL MST

print("\n" + "=" * 70)
print("3. CONSTRUCCIÓN DEL MST")
print("=" * 70)

distances = pdist(embeddings, metric='euclidean')
distance_matrix = squareform(distances)

print(f"✅ Matriz de distancias: {distance_matrix.shape}")
print(f"   Distancia mínima: {distance_matrix[distance_matrix > 0].min():.4f}")
print(f"   Distancia máxima: {distance_matrix.max():.4f}")
print(f"   Distancia promedio: {distance_matrix[distance_matrix > 0].mean():.4f}")

sparse_graph = csr_matrix(distance_matrix)
mst = minimum_spanning_tree(sparse_graph)
edge_lengths = mst.data
edge_lengths_sorted = np.sort(edge_lengths)[::-1]

print(f"\n✅ MST construido")
print(f"   Aristas: {len(edge_lengths)}")
print(f"   Longitud total: {edge_lengths.sum():.2f}")
print(f"   Longitud media: {edge_lengths.mean():.4f}")
print(f"   Longitud máxima: {edge_lengths.max():.4f}")


#SPECTRAL CLUSTERING CON REGULARIZACIÓN TEMPORAL


def spectral_clustering_temporal(embeddings, dates, n_clusters=None,
                                 sigma_spatial=None, tau_temporal=None,
                                 auto_tune=True):
    """
    Spectral Clustering con regularización temporal.

    W_ij = exp(-||x_i - x_j||^2 / σ^2) * exp(-|t_i - t_j| / τ)
    """
    n_samples = len(embeddings)

    dates_numeric = np.array([(d - dates.min()).days for d in dates])

    if sigma_spatial is None or auto_tune:
        distances_spatial = pdist(embeddings, metric='euclidean')
        sigma_spatial = np.median(distances_spatial) * 0.5
        print(f"   σ (espacial): {sigma_spatial:.4f} (auto-ajustado)")

    if tau_temporal is None or auto_tune:
        distances_temporal = pdist(dates_numeric.reshape(-1, 1), metric='cityblock')
        tau_temporal = np.median(distances_temporal) * 0.3
        print(f"   τ (temporal): {tau_temporal:.2f} días (auto-ajustado)")

    print(f"   Construyendo matriz de afinidad ({n_samples}x{n_samples})...")

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
        print(f"   Número de clusters estimado: {n_clusters} (eigengap heuristic)")

    print(f"   Aplicando Spectral Clustering con {n_clusters} clusters...")
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

print("\n" + "=" * 70)
print("3.5. SPECTRAL CLUSTERING CON REGULARIZACIÓN TEMPORAL")
print("=" * 70)

spectral_labels, spectral_affinity, spectral_params = spectral_clustering_temporal(
    embeddings, Z_index, n_clusters=None, auto_tune=True
)

print(f"\n✅ Spectral Clustering completado")
print(f"   Clusters detectados: {len(set(spectral_labels))}")
print(f"   Distribución: {dict(Counter(spectral_labels))}")

regimes = spectral_labels
use_spectral = True


#DETECCIÓN DE REGÍMENES


print("\n" + "=" * 70)
print("4. DETECCIÓN DE REGÍMENES")
print("=" * 70)

print("📊 Usando Spectral Clustering con regularización temporal")
n_regimes = len(set(regimes))
print(f"✅ Regímenes detectados: {n_regimes}")

regime_series = pd.Series(regimes, index=Z_index, name='regime')
y_aligned = y_series.reindex(Z_index)


#VALIDACIÓN CON MARKOV SWITCHING

print("\n" + "=" * 70)
print("4.5 VALIDACIÓN CON MARKOV SWITCHING")
print("=" * 70)

VALIDATE_MARKOV = True

if VALIDATE_MARKOV:
    try:
        k_regimes_ms = min(max(2, n_regimes), 5)
        print(f"📈 Ajustando MarkovRegression con {k_regimes_ms} regímenes...")
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

        min_len = min(len(regimes), len(markov_regimes))
        agreement = (regimes[:min_len] == markov_regimes[:min_len]).sum() / min_len * 100

        print(f"✅ Markov Switching ajustado")
        print(f"   Regímenes (Markov): {len(set(markov_regimes))}")
        print(f"   Acuerdo con Spectral: {agreement:.1f}%")
    except Exception as e:
        print("⚠️ No se pudo ajustar Markov Switching")
        print(f"   Detalle: {str(e)[:100]}")
        markov_regimes = None


# CARACTERÍSTICAS DE REGÍMENES


print("\n" + "=" * 70)
print("5. CARACTERÍSTICAS DE REGÍMENES (SIN INTERPRETACIÓN ECONÓMICA)")
print("=" * 70)

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

    economic_label = f"Régimen {new_regime_id + 1}"
    color = color_map(new_regime_id % 10)

    regime_interpretations[new_regime_id] = {
        'label': economic_label,
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

    print(f"\n{economic_label}:")
    print(f"  Duración: {duration} meses")
    print(f"  Período: {regime_dates.min().strftime('%Y-%m')} → {regime_dates.max().strftime('%Y-%m')}")
    print(f"  Crecimiento YoY promedio: {mean_growth:+.2f}% ± {std_growth:.2f} pp")
    print(f"  Rango: [{min_growth:.2f}%, {max_growth:.2f}%]")



