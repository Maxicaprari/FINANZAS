
# ANÁLISIS AVANZADO DE MERCADOS FINANCIEROS
# PCA + Clustering + Métricas de Red + Backtesting

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
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, mahalanobis
from scipy.stats import jarque_bera
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
import requests

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly no disponible. Se omitirán gráficos interactivos.")

# Configuración de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

print("✓ Librerías importadas correctamente\n")


# CONFIGURACIÓN DE PARÁMETROS
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
SOURCE = "yfinance"  # "yfinance" o "api"
START = "2010-01-01"
END = "2024-12-31"

API_ENDPOINT = "https://TU_API/precios"
API_HEADERS = {"Authorization": "Bearer TU_TOKEN"}
API_PARAMS = {"fecha_desde": START, "fecha_hasta": END}

WINDOW_SIZE = 60
STEP = 5
CRISIS_FUTURE_WINDOW = 60
CRISIS_THRESHOLD = -0.10
CRISIS_MAHALANOBIS_THRESHOLD = 2.5

N_CLUSTERS = 3
RANDOM_STATE = 42
RISK_FREE_RATE = 0.02

print(f"{'='*70}")
print(f"CONFIGURACIÓN DEL ANÁLISIS")
print(f"{'='*70}")
print(f"Período: {START} a {END}")
print(f"Tickers: {len(TICKERS)}")
print(f"Ventana: {WINDOW_SIZE} días, paso: {STEP}")
print(f"Clusters: {N_CLUSTERS}\n")

# FUNCIONES DE DESCARGA DE DATOS
def fetch_from_yfinance(tickers, start, end):
    import yfinance as yf
    print("📥 Descargando datos desde yfinance...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    print(f"✓ Descargados {data.shape[1]} tickers y {data.shape[0]} filas.")
    return data

def fetch_from_api(endpoint, headers=None, params=None):
    print("📥 Descargando datos desde API...")
    r = requests.get(endpoint, headers=headers, params=params, timeout=60)
    r.raise_for_status()
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
    raw['fecha'] = pd.to_datetime(raw['fecha'])
    data = raw.pivot(index="fecha", columns="ticker", values="precio_cierre").sort_index()
    print(f"✓ API -> matriz: {data.shape[0]} filas x {data.shape[1]} columnas.")
    return data

# FUNCIONES DE ANÁLISIS AVANZADO

def calculate_advanced_metrics(window_returns):
    """Calcula métricas de riesgo avanzadas por ventana"""
    metrics = {}
    wr = window_returns.dropna()

    if len(wr) < 2:
        return {k: np.nan for k in ['VaR_95', 'VaR_99', 'CVaR_95', 'CVaR_99',
                                     'max_drawdown', 'sortino_ratio', 'calmar_ratio']}

    # Value at Risk (VaR)
    metrics['VaR_95'] = np.percentile(wr, 5)
    metrics['VaR_99'] = np.percentile(wr, 1)

    # Conditional VaR (Expected Shortfall)
    metrics['CVaR_95'] = wr[wr <= metrics['VaR_95']].mean() if (wr <= metrics['VaR_95']).any() else metrics['VaR_95']
    metrics['CVaR_99'] = wr[wr <= metrics['VaR_99']].mean() if (wr <= metrics['VaR_99']).any() else metrics['VaR_99']

    # Maximum Drawdown
    cumulative = (1 + wr).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()

    # Sortino Ratio
    downside_returns = wr[wr < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    metrics['sortino_ratio'] = (wr.mean() * 252 - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0

    # Calmar Ratio
    metrics['calmar_ratio'] = (wr.mean() * 252) / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0

    return metrics

def detect_crisis_windows(pca_projections, threshold=2.5):
    """Detecta ventanas anómales usando distancia de Mahalanobis"""
    center = np.mean(pca_projections, axis=0)
    cov = np.cov(pca_projections.T)
    cov = cov + np.eye(cov.shape[0]) * 1e-6  # Regularización
    cov_inv = np.linalg.inv(cov)

    distances = []
    for point in pca_projections:
        dist = mahalanobis(point, center, cov_inv)
        distances.append(dist)

    distances = np.array(distances)
    crisis_mask = distances > threshold

    return crisis_mask, distances

def test_stationarity(timeseries, name="Series"):
    """Test de Dickey-Fuller para estacionariedad"""
    try:
        result = adfuller(timeseries.dropna())
        return {
            'name': name,
            'adf_stat': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    except:
        return {'name': name, 'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False}

def test_normality(returns):
    """Tests de normalidad en retornos"""
    results = {}
    for ticker in returns.columns:
        data = returns[ticker].dropna()
        if len(data) > 0:
            jb_stat, jb_pval = jarque_bera(data)
            results[ticker] = {
                'JB_statistic': jb_stat,
                'p_value': jb_pval,
                'normal': jb_pval > 0.05,
                'skewness': data.skew(),
                'kurtosis': data.kurtosis()
            }
    return pd.DataFrame(results).T

def risk_contribution_analysis(returns_data, weights=None):
    """Calcula la contribución marginal al riesgo de cada activo"""
    if weights is None:
        weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)

    cov_matrix = returns_data.cov() * 252
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights / portfolio_vol
    risk_contrib = weights * marginal_contrib
    pct_contrib = risk_contrib / risk_contrib.sum()

    results = pd.DataFrame({
        'peso': weights,
        'contrib_marginal': marginal_contrib,
        'contrib_riesgo': risk_contrib,
        'pct_contrib_riesgo': pct_contrib
    }, index=returns_data.columns)

    return results.sort_values('pct_contrib_riesgo', ascending=False)

def generate_regime_signals(pca_projections, clusters, returns_data, window_dates):
    """Genera señales de trading basadas en cambios de régimen"""
    signals = pd.DataFrame(index=returns_data.index)

    # Definir estrategia por régimen
    regime_allocations = {
        0: 0.5,   # 50% invertido
        1: 1.0,   # 100% invertido
        2: 0.0    # 0% invertido (crisis)
    }

    # Identificar régimen según PC1
    pc1_means = [pca_projections[clusters == i, 0].mean() for i in range(N_CLUSTERS)]
    regime_mapping = {
        np.argmin(pc1_means): 2,  # PC1 más bajo = crisis
        np.argmax(pc1_means): 1,  # PC1 más alto = expansión
    }
    for i in range(N_CLUSTERS):
        if i not in regime_mapping:
            regime_mapping[i] = 0

    mapped_clusters = np.array([regime_mapping[c] for c in clusters])

    signals['regime'] = mapped_clusters[0]
    signals['allocation'] = regime_allocations[mapped_clusters[0]]

    for i, cluster in enumerate(mapped_clusters):
        if i < len(window_dates):
            date = window_dates[i]
            if date in signals.index:
                signals.loc[date:, 'regime'] = cluster
                signals.loc[date:, 'allocation'] = regime_allocations[cluster]

    signals = signals.fillna(method='ffill').fillna(0.5)
    return signals

def backtest_regime_strategy(returns_data, signals, initial_capital=100000):
    """Backtest de estrategia basada en regímenes"""
    portfolio_value = [initial_capital]

    common_index = returns_data.index.intersection(signals.index)
    returns_aligned = returns_data.loc[common_index]
    signals_aligned = signals.loc[common_index]

    for i in range(1, len(returns_aligned)):
        allocation = signals_aligned.iloc[i]['allocation']
        daily_return = returns_aligned.iloc[i].mean()
        pnl = portfolio_value[-1] * allocation * daily_return
        portfolio_value.append(portfolio_value[-1] + pnl)

    portfolio_series = pd.Series(portfolio_value, index=common_index)

    total_return = (portfolio_value[-1] / initial_capital - 1)
    returns_pct = portfolio_series.pct_change().dropna()
    sharpe = returns_pct.mean() / returns_pct.std() * np.sqrt(252) if returns_pct.std() > 0 else 0

    cumulative = portfolio_series / initial_capital
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        'portfolio_series': portfolio_series,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': portfolio_value[-1]
    }

# DESCARGA DE DATOS
if SOURCE == "yfinance":
    prices = fetch_from_yfinance(TICKERS, START, END)
else:
    prices = fetch_from_api(API_ENDPOINT, headers=API_HEADERS, params=API_PARAMS)

prices = prices.sort_index()
prices = prices.loc[~prices.index.duplicated(keep="first")]
prices = prices.dropna(axis=1, thresh=len(prices)*0.80)
prices = prices.fillna(method='ffill').fillna(method='bfill')
prices = prices.dropna(axis=0)

TICKERS_FINAL = list(prices.columns)
print(f"\n✓ Tickers finales: {len(TICKERS_FINAL)}")
print(f"✓ Período efectivo: {prices.index[0]} a {prices.index[-1]}")
print(f"✓ Observaciones: {len(prices)}\n")

# CÁLCULO DE RETORNOS

log_returns = np.log(prices / prices.shift(1)).dropna()

print(f"{'='*70}")
print("ESTADÍSTICAS DESCRIPTIVAS (Anualizadas)")
print(f"{'='*70}")
print(f"Retorno medio: {(log_returns.mean() * 252).mean():.2%}")
print(f"Volatilidad media: {(log_returns.std() * np.sqrt(252)).mean():.2%}\n")

# FUNCIONES DE MÉTRICAS ROLLING (MEJORADAS)

def rolling_metrics_enhanced(returns, window=60, step=5):
    """Métricas rolling con métricas avanzadas"""
    rows = []
    idx = returns.index
    print(f"📊 Calculando métricas rolling (ventana={window}, step={step})...")

    for start in range(0, len(idx) - window, step):
        end = start + window
        win = returns.iloc[start:end]
        end_date = idx[end-1]

        # Métricas básicas
        mu = win.mean() * 252
        vol = win.std() * np.sqrt(252)
        sharpe = (mu - RISK_FREE_RATE) / vol.replace(0, np.nan)
        ret_cum = win.sum()

        # Métricas avanzadas por activo
        for ticker in win.columns:
            adv_metrics = calculate_advanced_metrics(win[ticker])
            tmp = {
                "end_date": end_date,
                "ticker": ticker,
                "ret_log_cum_window": ret_cum[ticker],
                "mu_ann": mu[ticker],
                "vol_ann": vol[ticker],
                "sharpe": sharpe[ticker],
                **adv_metrics
            }
            rows.append(tmp)

    out = pd.DataFrame(rows)
    out.set_index(["end_date", "ticker"], inplace=True)
    print("✓ Métricas rolling calculadas\n")
    return out

def extract_network_metrics_enhanced(log_returns, window_size=60, step=5):
    """Métricas de red mejoradas"""
    metrics = []
    idx = log_returns.index
    print(f"🕸️  Extrayendo métricas de red (ventana={window_size}, step={step})...")

    for start in range(0, len(idx)-window_size, step):
        end = start + window_size
        end_date = idx[end-1]
        w = log_returns.iloc[start:end]
        corr = w.corr()
        dist = np.sqrt(2*(1-corr)).fillna(0.0)

        G = nx.from_pandas_adjacency(dist)
        avg_edge_weight_full = (G.size(weight='weight') / G.number_of_edges()) if G.number_of_edges() > 0 else 0.0
        avg_clustering_weighted = nx.average_clustering(G, weight='weight')

        mst = nx.minimum_spanning_tree(G)
        mst_total_weight = mst.size(weight='weight')
        try:
            mst_avg_shortest_path = nx.average_shortest_path_length(mst, weight='weight')
        except nx.NetworkXError:
            mst_avg_shortest_path = np.nan

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
    print("✓ Métricas de red calculadas\n")
    return df

# Ejecutar análisis de ventanas
rolling_metrics_df = rolling_metrics_enhanced(log_returns, window=WINDOW_SIZE, step=STEP)
network_metrics_df = extract_network_metrics_enhanced(log_returns, window_size=WINDOW_SIZE, step=STEP)


# PCA + KMEANS (CON PROYECCIONES TEMPORALES)

print(f"{'='*70}")
print("PCA Y CLUSTERING")
print(f"{'='*70}")

X_scaled = StandardScaler().fit_transform(log_returns.fillna(0))
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

var_exp = pca_full.explained_variance_ratio_
var_exp_cum = np.cumsum(var_exp)

print(f"✓ PC1 explica: {var_exp[0]:.2%}")
print(f"✓ PC2 explica: {var_exp[1]:.2%}")
print(f"✓ PC1+PC2 explican: {var_exp_cum[1]:.2%}")
print(f"✓ Primeros 5 componentes: {var_exp_cum[4]:.2%}\n")

# PCA sobre activos (transpose)
def pca_kmeans_on_assets(returns_df, n_clusters=3, random_state=42):
    X = returns_df.T.values
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)
    pca_df = pd.DataFrame(Z, index=returns_df.columns, columns=["PC1", "PC2"])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    pca_df["Cluster"] = km.fit_predict(pca_df[["PC1","PC2"]])

    return pca_df, pca

pca_clusters_df, pca_asset_model = pca_kmeans_on_assets(
    log_returns, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
)

print(f"✓ Clustering de activos completado")
print(f"  Distribución de clusters:")
for c in range(N_CLUSTERS):
    count = (pca_clusters_df['Cluster'] == c).sum()
    print(f"    Cluster {c}: {count} activos")

# PROYECCIÓN TEMPORAL EN VENTANAS
print(f"\n{'='*70}")
print("PROYECCIÓN TEMPORAL DE VENTANAS")
print(f"{'='*70}")

scaler_temporal = StandardScaler()
window_projections = []
window_dates = []

for start in range(0, len(log_returns) - WINDOW_SIZE, STEP):
    end = start + WINDOW_SIZE
    window = log_returns.iloc[start:end]
    window_date = window.index[-1]

    window_scaled = scaler_temporal.fit_transform(window.fillna(0))
    window_pca = pca_full.transform(window_scaled).mean(axis=0)[:2]
    window_projections.append(window_pca)
    window_dates.append(window_date)

window_projections = np.array(window_projections)

# KMeans sobre ventanas
kmeans_windows = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
clusters = kmeans_windows.fit_predict(window_projections)

print(f"✓ {len(window_dates)} ventanas proyectadas")
print(f"  Distribución de regímenes:")
for c in range(N_CLUSTERS):
    count = (clusters == c).sum()
    print(f"    Régimen {c}: {count} ventanas ({count/len(clusters)*100:.1f}%)")


# DETECCIÓN DE CRISIS
print(f"\n{'='*70}")
print("DETECCIÓN DE CRISIS")
print(f"{'='*70}")

crisis_windows, distances = detect_crisis_windows(
    window_projections, threshold=CRISIS_MAHALANOBIS_THRESHOLD
)

print(f"✓ Ventanas de crisis detectadas: {crisis_windows.sum()} ({crisis_windows.sum()/len(window_dates)*100:.1f}%)")

if crisis_windows.sum() > 0:
    crisis_dates_list = np.array(window_dates)[crisis_windows]
    print(f"\n📅 Fechas críticas (primeras 10):")
    for date in crisis_dates_list[:10]:
        print(f"    - {date.strftime('%Y-%m-%d')}")

# TESTS ESTADÍSTICOS

print(f"\n{'='*70}")
print("TESTS ESTADÍSTICOS")
print(f"{'='*70}")

# Estacionariedad (muestra de 5 activos)
print("\n🔍 Test de Estacionariedad (muestra):")
for ticker in TICKERS_FINAL[:5]:
    result = test_stationarity(log_returns[ticker], name=ticker)
    status = "✓ Estacionaria" if result['is_stationary'] else "✗ No estacionaria"
    print(f"  {ticker}: {status} (p={result['p_value']:.4f})")

# Normalidad
print("\n🔍 Test de Normalidad (Jarque-Bera):")
normality_results = test_normality(log_returns)
normal_count = normality_results['normal'].sum()
print(f"  Activos con distribución normal: {normal_count}/{len(normality_results)} ({normal_count/len(normality_results)*100:.1f}%)")


# ANÁLISIS DE RIESGO

print(f"\n{'='*70}")
print("ANÁLISIS DE CONTRIBUCIÓN AL RIESGO")
print(f"{'='*70}")

risk_contrib = risk_contribution_analysis(log_returns)

print(f"\n📊 Top 10 contribuyentes al riesgo del portafolio:")
print(risk_contrib.head(10)[['pct_contrib_riesgo']])


# BACKTESTING

print(f"\n{'='*70}")
print("BACKTESTING DE ESTRATEGIA POR REGÍMENES")
print(f"{'='*70}")

signals = generate_regime_signals(window_projections, clusters, log_returns, window_dates)
backtest_results = backtest_regime_strategy(log_returns, signals, initial_capital=100000)

# Buy & Hold para comparación
bnh_series = (1 + log_returns.mean(axis=1)).cumprod() * 100000
bnh_return = (bnh_series.iloc[-1] / 100000) - 1

print(f"\n✓ Resultados del Backtest:")
print(f"  Capital inicial: $100,000")
print(f"  Capital final: ${backtest_results['final_value']:,.2f}")
print(f"  Retorno total: {backtest_results['total_return']:.2%}")
print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
print(f"\n  Buy & Hold retorno: {bnh_return:.2%}")
print(f"  Alpha generado: {(backtest_results['total_return'] - bnh_return):.2%}")


# VISUALIZACIONES

print(f"\n{'='*70}")
print("GENERANDO VISUALIZACIONES")
print(f"{'='*70}")

sns.set_style("whitegrid")
sns.set_context("talk")

# 1. Clusters de activos (PCA sobre activos transpose)
plt.figure(figsize=(12,8))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_clusters_df,
                s=200, palette="Set2", edgecolor='black', linewidth=1.5)
for t in pca_clusters_df.index:
    plt.text(pca_clusters_df.loc[t,"PC1"]+0.01, pca_clusters_df.loc[t,"PC2"]+0.01,
             t, fontsize=9, fontweight='bold')
plt.xlabel("PC1", fontsize=12, fontweight='bold')
plt.ylabel("PC2", fontsize=12, fontweight='bold')
plt.title("Clusters de Activos (PCA + KMeans)", fontsize=14, fontweight="bold")
plt.legend(title="Cluster", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Scree Plot
plt.figure(figsize=(10, 6))
var_ratio = var_exp[:10] * 100
cum_var = var_exp_cum[:10] * 100

bars = plt.bar(range(1, len(var_ratio)+1), var_ratio,
               alpha=0.7, color=sns.color_palette("Spectral", len(var_ratio)),
               label="Varianza individual")
plt.plot(range(1, len(var_ratio)+1), cum_var, marker='o', color="black", lw=2,
         label="Varianza acumulada")

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{var_ratio[i]:.1f}%", ha='center', fontsize=9, color="dimgray")

plt.xlabel("Número de Componentes", fontsize=12, fontweight='bold')
plt.ylabel("Varianza Explicada (%)", fontsize=12, fontweight='bold')
plt.title("Scree Plot - Varianza Explicada", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Biplot mejorado
def biplot_enhanced(scores, coeffs, labels=None):
    xs, ys = scores[:, 0], scores[:, 1]
    plt.figure(figsize=(14, 10))
    plt.scatter(xs, ys, alpha=0.6, s=40, c=np.arange(len(xs)), cmap="viridis", edgecolor='k', linewidth=0.3)

    for i in range(coeffs.shape[0]):
        plt.arrow(0, 0, coeffs[i, 0]*max(xs)*0.8, coeffs[i, 1]*max(ys)*0.8,
                  color='darkslateblue', alpha=0.7, head_width=0.15, linewidth=2)
        name = labels[i] if labels is not None else f"Var{i+1}"
        plt.text(coeffs[i, 0]*max(xs)*0.9,
                 coeffs[i, 1]*max(ys)*0.9,
                 name, color='darkslateblue', fontsize=10, fontweight='bold')

    plt.xlabel("PC1", fontsize=12, fontweight='bold')
    plt.ylabel("PC2", fontsize=12, fontweight='bold')
    plt.title("Biplot (PC1 vs PC2) - Cargas de Variables", fontsize=14, fontweight="bold")
    plt.colorbar(label="Índice temporal")
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

biplot_enhanced(X_pca_full, pca_full.components_.T, labels=log_returns.columns)

# 4. Heatmap de correlaciones
plt.figure(figsize=(16, 14))
sns.heatmap(log_returns.corr(), annot=True, fmt=".2f", cmap="crest",
            annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8})
plt.title("Matriz de Correlación entre Retornos", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 5. Proyección temporal de ventanas con crisis marcadas
plt.figure(figsize=(12, 9))
scatter = plt.scatter(window_projections[:, 0], window_projections[:, 1],
                     c=clusters, cmap="Set2", alpha=0.7, s=80,
                     edgecolor="k", linewidth=0.5)
plt.plot(window_projections[:, 0], window_projections[:, 1],
         color="gray", linewidth=0.6, alpha=0.4)

# Marcar crisis
if crisis_windows.sum() > 0:
    plt.scatter(window_projections[crisis_windows, 0],
               window_projections[crisis_windows, 1],
               c='red', marker='*', s=500, edgecolors='black',
               linewidths=1.5, alpha=0.8, label='Crisis', zorder=10)

# Centroides
plt.scatter(kmeans_windows.cluster_centers_[:, 0],
           kmeans_windows.cluster_centers_[:, 1],
           marker="X", s=320, c="black", label="Centroides",
           edgecolor="white", linewidth=2, zorder=10)

# Anotar fechas extremas
dist = np.sqrt(window_projections[:, 0]**2 + window_projections[:, 1]**2)
top_idx = dist.argsort()[-10:]
for i in top_idx:
    plt.text(window_projections[i, 0], window_projections[i, 1],
             window_dates[i].strftime('%Y-%m-%d'),
             fontsize=8, color="darkslategray")

plt.xlabel(f"PC1 ({var_exp[0]:.1%} varianza)", fontsize=12, fontweight='bold')
plt.ylabel(f"PC2 ({var_exp[1]:.1%} varianza)", fontsize=12, fontweight='bold')
plt.title("Proyección Temporal de Ventanas en Espacio PCA\nRegímenes de Mercado",
         fontsize=14, fontweight="bold")
plt.colorbar(scatter, label="Régimen")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Serie temporal PC1
fig, ax = plt.subplots(figsize=(16, 6))
pc1_series = pd.Series(window_projections[:, 0], index=window_dates)
ax.plot(pc1_series.index, pc1_series.values, linewidth=2, color='navy')
ax.fill_between(pc1_series.index, pc1_series.values, 0, alpha=0.3)

if crisis_windows.sum() > 0:
    crisis_dates_arr = np.array(window_dates)[crisis_windows]
    crisis_pc1 = pc1_series[crisis_dates_arr]
    ax.scatter(crisis_pc1.index, crisis_pc1.values, c='red', s=150,
              zorder=5, label='Crisis', marker='*', edgecolors='black', linewidth=1.5)

ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
ax.set_ylabel('PC1', fontsize=12, fontweight='bold')
ax.set_title('Evolución Temporal del Primer Componente Principal (PC1)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Dendrograma jerárquico
plt.figure(figsize=(16, 8))
corr = log_returns.corr()
dist = 1 - corr
dist_vec = squareform(dist.values, checks=False)
Z = linkage(dist_vec, method="ward")
dendrogram(Z, labels=corr.columns, leaf_rotation=90, leaf_font_size=10)
plt.title("Dendrograma Jerárquico de Activos (1 - correlación)",
         fontsize=14, fontweight="bold")
plt.xlabel("Ticker", fontsize=12, fontweight='bold')
plt.ylabel("Distancia", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 8. Performance del Backtest
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Curva de capital
ax1.plot(backtest_results['portfolio_series'].index,
        backtest_results['portfolio_series'].values,
        linewidth=2, color='green', label='Estrategia por Regímenes')

bnh_aligned = bnh_series[bnh_series.index.isin(backtest_results['portfolio_series'].index)]
ax1.plot(bnh_aligned.index, bnh_aligned.values,
        linewidth=2, color='blue', alpha=0.7, label='Buy & Hold', linestyle='--')

ax1.set_xlabel('Fecha', fontsize=12, fontweight='bold')
ax1.set_ylabel('Valor del Portafolio ($)', fontsize=12, fontweight='bold')
ax1.set_title('Performance: Estrategia vs Buy & Hold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')

# Drawdown
portfolio_series = backtest_results['portfolio_series']
cumulative = portfolio_series / portfolio_series.iloc[0]
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max

ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                color='red', alpha=0.3)
ax2.plot(drawdown.index, drawdown.values * 100, linewidth=2, color='darkred')
ax2.set_xlabel('Fecha', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown de la Estrategia', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Métricas de red a lo largo del tiempo
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

network_metrics_df.plot(y='avg_corr_offdiag', ax=axes[0,0], color='steelblue', linewidth=2)
axes[0,0].set_title('Correlación Promedio', fontweight='bold')
axes[0,0].grid(alpha=0.3)

network_metrics_df.plot(y='mst_total_weight', ax=axes[0,1], color='darkgreen', linewidth=2)
axes[0,1].set_title('Peso Total MST', fontweight='bold')
axes[0,1].grid(alpha=0.3)

network_metrics_df.plot(y='avg_clustering_weighted', ax=axes[1,0], color='darkorange', linewidth=2)
axes[1,0].set_title('Clustering Promedio', fontweight='bold')
axes[1,0].grid(alpha=0.3)

network_metrics_df.plot(y='mst_avg_shortest_path', ax=axes[1,1], color='purple', linewidth=2)
axes[1,1].set_title('Camino Promedio MST', fontweight='bold')
axes[1,1].grid(alpha=0.3)

plt.suptitle('Evolución de Métricas de Red', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

print("✓ Visualizaciones estáticas generadas\n")


# DASHBOARD INTERACTIVO CON PLOTLY

if PLOTLY_AVAILABLE:
    print(f"{'='*70}")
    print("CREANDO DASHBOARD INTERACTIVO")
    print(f"{'='*70}")

    fig_interactive = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PCA - Regímenes de Mercado',
                       'Evolución Temporal PC1',
                       'Heatmap de Correlaciones (Top 15)',
                       'Performance Backtest'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )

    # 1. Scatter PCA con clusters
    for cluster in range(N_CLUSTERS):
        mask = clusters == cluster
        fig_interactive.add_trace(
            go.Scatter(
                x=window_projections[mask, 0],
                y=window_projections[mask, 1],
                mode='markers',
                name=f'Régimen {cluster}',
                marker=dict(size=8),
                text=[d.strftime('%Y-%m-%d') for d in np.array(window_dates)[mask]],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}'
            ),
            row=1, col=1
        )

    # Marcar crisis
    if crisis_windows.sum() > 0:
        fig_interactive.add_trace(
            go.Scatter(
                x=window_projections[crisis_windows, 0],
                y=window_projections[crisis_windows, 1],
                mode='markers',
                name='Crisis',
                marker=dict(size=15, symbol='star', color='red'),
                text=[d.strftime('%Y-%m-%d') for d in np.array(window_dates)[crisis_windows]],
                hovertemplate='<b>CRISIS: %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}'
            ),
            row=1, col=1
        )

    # 2. Serie temporal PC1
    fig_interactive.add_trace(
        go.Scatter(
            x=window_dates,
            y=window_projections[:, 0],
            mode='lines',
            name='PC1',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>PC1: %{y:.2f}'
        ),
        row=1, col=2
    )

    # 3. Heatmap correlaciones (top 15)
    top_15_tickers = risk_contrib.head(15).index
    corr_top15 = log_returns[top_15_tickers].corr()
    fig_interactive.add_trace(
        go.Heatmap(
            z=corr_top15.values,
            x=corr_top15.columns,
            y=corr_top15.columns,
            colorscale='RdBu',
            zmid=0,
            showscale=True
        ),
        row=2, col=1
    )

    # 4. Performance backtest
    fig_interactive.add_trace(
        go.Scatter(
            x=backtest_results['portfolio_series'].index,
            y=backtest_results['portfolio_series'].values,
            mode='lines',
            name='Estrategia',
            line=dict(color='green', width=2),
            hovertemplate='%{x}<br>Valor: $%{y:,.0f}'
        ),
        row=2, col=2
    )

    fig_interactive.add_trace(
        go.Scatter(
            x=bnh_aligned.index,
            y=bnh_aligned.values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='blue', width=2, dash='dash'),
            hovertemplate='%{x}<br>Valor: $%{y:,.0f}'
        ),
        row=2, col=2
    )

    fig_interactive.update_layout(
        height=900,
        showlegend=True,
        title_text="Dashboard Interactivo - Análisis de Mercados Financieros"
    )

    fig_interactive.show()
    print("✓ Dashboard interactivo generado\n")


# RESUMEN FINAL
print(f"\n{'='*70}")
print("RESUMEN FINAL DEL ANÁLISIS")
print(f"{'='*70}")
