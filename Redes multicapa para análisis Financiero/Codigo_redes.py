import yfinance as yf
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# === CONFIGURACIÓN INICIAL ===
# Removidos YPFD y PAMP (deslistados)
tickers = [
    'GGAL', 'LOMA', 'BMA', 'MELI', 'IRS', 'CRESY',
    '^MERV', '^GSPC', 'EWZ',
    'ZS=F', 'ZC=F', 'CL=F', 'GC=F',
    '^TNX'
]

print("🚀 Iniciando Análisis Multicapa de Redes Financieras")
print("=" * 60)

# === PASO 1: DESCARGAR Y PREPARAR DATOS ===
print("\n Descargando datos financieros...")

try:
    # Descargar datos completos
    raw_data = yf.download(tickers, start="2018-01-01", end="2025-01-01", auto_adjust=True)
    
    if raw_data.empty:
        print("❌ No se pudieron descargar datos")
        exit()
    
    # Extraer diferentes tipos de datos con validación
    close_data = raw_data['Close'].dropna(axis=1, how='all')
    
    # Filtrar columnas con suficientes datos (al menos 80% de observaciones)
    min_data_points = int(len(close_data) * 0.8)
    close_data = close_data.dropna(axis=1, thresh=min_data_points)
    
    if close_data.empty or len(close_data.columns) < 3:
        print("❌ Datos insuficientes después del filtrado")
        exit()
    
    # Verificar que tenemos datos de volumen
    try:
        volume_data = raw_data['Volume'].dropna(axis=1, how='all')
        volume_data = volume_data.dropna(axis=1, thresh=min_data_points)
        
        high_data = raw_data['High'].dropna(axis=1, how='all')
        high_data = high_data.dropna(axis=1, thresh=min_data_points)
        
        low_data = raw_data['Low'].dropna(axis=1, how='all')
        low_data = low_data.dropna(axis=1, thresh=min_data_points)
        
        has_volume_data = not volume_data.empty and not high_data.empty and not low_data.empty
        print(" Datos de volumen y precios disponibles" if has_volume_data else "⚠️  Datos de volumen limitados")
    except:
        has_volume_data = False
        print("⚠️  Algunos datos de volumen no disponibles")
    
    # Calcular métricas para cada capa
    log_returns = np.log(close_data / close_data.shift(1)).dropna()
    
    if log_returns.empty or len(log_returns) < 100:
        print(f"❌ Datos insuficientes para análisis ({len(log_returns)} observaciones)")
        exit()
    
    volatility = log_returns.rolling(window=20).std().dropna()
    
    if has_volume_data:
        # Filtrar volumen para que coincida con close_data
        common_tickers = list(set(close_data.columns) & set(volume_data.columns))
        if common_tickers:
            volume_changes = np.log(volume_data[common_tickers] / volume_data[common_tickers].shift(1)).dropna()
            high_low_range = np.log(high_data[common_tickers] / low_data[common_tickers]).dropna()
        else:
            has_volume_data = False
    
    print(f" Datos procesados. Activos disponibles: {len(close_data.columns)}")
    print(f" Período: {log_returns.index[0].date()} a {log_returns.index[-1].date()}")
    print(f" Total observaciones: {len(log_returns)}")
    
except Exception as e:
    print(f" Error descargando datos: {e}")
    exit()

# === PASO 2: DEFINIR MAPEO SECTORIAL ===
sector_mapping = {
    # Financiero
    'GGAL': 'financiero', 'BMA': 'financiero',
    # Materiales
    'LOMA': 'materiales',
    # Tecnología/Servicios
    'MELI': 'tecnologia', 'IRS': 'inmobiliario',
    # Agropecuario
    'CRESY': 'agropecuario',
    # Índices
    '^MERV': 'indice_local', '^GSPC': 'indice_global', 'EWZ': 'indice_regional',
    # Commodities
    'ZS=F': 'commodities', 'ZC=F': 'commodities', 'CL=F': 'commodities', 'GC=F': 'commodities',
    # Bonos
    '^TNX': 'bonos'
}

# === PASO 3: FUNCIONES PARA CONSTRUIR CAPAS ===
def build_correlation_layer(data, layer_name):
    """Construye una capa basada en correlaciones"""
    if data.empty or len(data.columns) < 2:
        return nx.Graph()
    
    try:
        corr_matrix = data.corr()
        
        # Verificar que no hay NaNs
        if corr_matrix.isnull().all().all():
            return nx.Graph()
        
        # Llenar NaNs con 0 para el cálculo de distancia
        corr_matrix = corr_matrix.fillna(0)
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        G = nx.Graph()
        G.name = layer_name
        
        for i in dist_matrix.columns:
            for j in dist_matrix.columns:
                if i != j and not np.isnan(dist_matrix.loc[i, j]):
                    G.add_edge(i, j, weight=dist_matrix.loc[i, j], correlation=corr_matrix.loc[i, j])
        
        return G
    except Exception as e:
        print(f"⚠️  Error construyendo capa {layer_name}: {e}")
        return nx.Graph()

def build_sector_layer(available_tickers):
    """Construye capa basada en sectores"""
    G = nx.Graph()
    G.name = 'sector'
    
    # Relacionar sectores
    related_sectors = {
        ('financiero', 'indice_local'): 0.3,
        ('agropecuario', 'commodities'): 0.3,
        ('indice_global', 'bonos'): 0.4,
        ('indice_regional', 'indice_local'): 0.2
    }
    
    for ticker1, ticker2 in combinations(available_tickers, 2):
        if ticker1 in sector_mapping and ticker2 in sector_mapping:
            sector1 = sector_mapping[ticker1]
            sector2 = sector_mapping[ticker2]
            
            if sector1 == sector2:
                weight = 0.1  # Mismo sector = alta conexión (distancia baja)
            else:
                # Buscar si hay relación entre sectores
                weight = 1.0  # Por defecto, sectores no relacionados
                for (s1, s2), rel_weight in related_sectors.items():
                    if (sector1, sector2) == (s1, s2) or (sector2, sector1) == (s1, s2):
                        weight = rel_weight
                        break
                        
            G.add_edge(ticker1, ticker2, weight=weight, sector1=sector1, sector2=sector2)
    
    return G

def extract_layer_metrics(layer_graph, layer_name):
    """Extrae métricas de una capa específica"""
    if len(layer_graph.nodes()) == 0:
        return {}
    
    try:
        # Calcular MST
        mst = nx.minimum_spanning_tree(layer_graph)
        
        if len(mst.nodes()) == 0:
            return {}
        
        # Métricas básicas
        metrics = {
            f'{layer_name}_avg_degree': np.mean([deg for _, deg in dict(mst.degree()).items()]) if mst.degree() else 0,
            f'{layer_name}_density': nx.density(mst),
            f'{layer_name}_avg_clustering': nx.average_clustering(mst),
            f'{layer_name}_num_edges': mst.number_of_edges(),
            f'{layer_name}_num_nodes': mst.number_of_nodes()
        }
        
        # Métricas que requieren conectividad
        if nx.is_connected(mst) and len(mst.nodes()) > 1:
            metrics[f'{layer_name}_avg_shortest_path'] = nx.average_shortest_path_length(mst)
            metrics[f'{layer_name}_diameter'] = nx.diameter(mst)
        else:
            metrics[f'{layer_name}_avg_shortest_path'] = np.nan
            metrics[f'{layer_name}_diameter'] = np.nan
        
        # Centralidad
        if len(layer_graph.nodes()) > 1:
            centrality = nx.betweenness_centrality(layer_graph)
            if centrality:
                metrics[f'{layer_name}_max_centrality'] = max(centrality.values())
                metrics[f'{layer_name}_centrality_std'] = np.std(list(centrality.values()))
            else:
                metrics[f'{layer_name}_max_centrality'] = 0
                metrics[f'{layer_name}_centrality_std'] = 0
        else:
            metrics[f'{layer_name}_max_centrality'] = 0
            metrics[f'{layer_name}_centrality_std'] = 0
            
        return metrics
    
    except Exception as e:
        print(f"  Error calculando métricas para capa {layer_name}: {e}")
        return {}

# === PASO 4: EXTRAER MÉTRICAS MULTICAPA ===
print("\n🔄 Extrayendo métricas multicapa...")

def extract_multilayer_metrics(window_size=60, step=10):
    """Extrae métricas de red multicapa usando ventanas deslizantes"""
    available_tickers = log_returns.columns.tolist()
    
    # VALIDACIÓN CRÍTICA
    if len(log_returns) < window_size:
        print(f"⚠️  No hay suficientes datos ({len(log_returns)}) para ventana de {window_size} días")
        return pd.DataFrame()
    
    total_windows = (len(log_returns) - window_size) // step
    if total_windows <= 0:
        print(f"⚠️  No se pueden crear ventanas con los datos disponibles")
        return pd.DataFrame()
    
    metrics_list = []
    print(f"Procesando {total_windows} ventanas de {window_size} días...")
    
    for i, start in enumerate(range(0, len(log_returns) - window_size, step)):
        if i % 20 == 0:
            print(f"Progreso: {i}/{total_windows} ventanas procesadas")
            
        end = start + window_size
        window_date = log_returns.index[end - 1]
        
        # Datos para esta ventana
        window_returns = log_returns.iloc[start:end]
        
        # Construir capas para esta ventana
        layers = {}
        
        # Capa 1: Retornos
        layers['returns'] = build_correlation_layer(window_returns, 'returns')
        
        # Capa 2: Volatilidad
        window_volatility = volatility.iloc[start:end] if len(volatility) > end else volatility.tail(window_size)
        if len(window_volatility.dropna(axis=1)) > 1:
            layers['volatility'] = build_correlation_layer(window_volatility.dropna(axis=1), 'volatility')
        
        # Capa 3: Volumen (si disponible)
        if has_volume_data and 'volume_changes' in locals() and len(volume_changes) > end:
            window_volume = volume_changes.iloc[start:end]
            if len(window_volume.dropna(axis=1)) > 1:
                layers['volume'] = build_correlation_layer(window_volume.dropna(axis=1), 'volume')
        
        # Capa 4: Alta frecuencia (si disponible)
        if has_volume_data and 'high_low_range' in locals() and len(high_low_range) > end:
            window_highlow = high_low_range.iloc[start:end]
            if len(window_highlow.dropna(axis=1)) > 1:
                layers['high_freq'] = build_correlation_layer(window_highlow.dropna(axis=1), 'high_freq')
        
        # Capa 5: Sectorial
        layers['sector'] = build_sector_layer(available_tickers)
        
        # Extraer métricas de cada capa
        layer_metrics = {'end_date': window_date}
        
        for layer_name, layer_graph in layers.items():
            layer_metrics.update(extract_layer_metrics(layer_graph, layer_name))
        
        # Métrica inter-capa
        if 'returns' in layers and 'volatility' in layers:
            try:
                edges1 = set(layers['returns'].edges())
                edges2 = set(layers['volatility'].edges())
                intersection = len(edges1.intersection(edges2))
                union = len(edges1.union(edges2))
                layer_metrics['interlayer_similarity'] = intersection / union if union > 0 else 0
            except:
                layer_metrics['interlayer_similarity'] = 0
        
        metrics_list.append(layer_metrics)
    
    # VALIDACIÓN ANTES DE CREAR DATAFRAME
    if not metrics_list:
        print("  No se pudieron extraer métricas")
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_list)
    
    # Verificar que la columna end_date existe
    if 'end_date' not in df.columns:
        print("  Error: columna end_date no encontrada")
        return df
    
    return df.set_index('end_date')

# Extraer métricas
multilayer_metrics = extract_multilayer_metrics()

if multilayer_metrics.empty:
    print("❌ No se pudieron extraer métricas. Datos insuficientes.")
    exit()

print(f" Métricas extraídas: {len(multilayer_metrics)} ventanas temporales")
print(f" Características disponibles: {len(multilayer_metrics.columns)}")

# === PASO 5: PREPARAR DATOS PARA PREDICCIÓN ===
print("\n Preparando modelo de predicción...")

# Definir variable objetivo (crisis)
target_ticker = '^MERV'
future_window = 30
crisis_threshold = -0.08

if target_ticker in log_returns.columns:
    future_returns = log_returns[target_ticker].rolling(window=future_window).sum().shift(-future_window)
    multilayer_metrics['future_return'] = future_returns
    multilayer_metrics['crisis'] = (multilayer_metrics['future_return'] < crisis_threshold).astype(int)
    multilayer_metrics.dropna(inplace=True)
    
    print(f" Crisis definida como retorno de {target_ticker} < {crisis_threshold*100}% en {future_window} días")
    print(f" Períodos de crisis identificados: {multilayer_metrics['crisis'].sum()}")
    print(f" Total de observaciones: {len(multilayer_metrics)}")
    
    if multilayer_metrics['crisis'].sum() > 0 and len(multilayer_metrics) > 50:
        # === PASO 6: ENTRENAR MODELOS ===
        print("\n Entrenando modelos de machine learning...")
        
        # Preparar características
        feature_columns = [col for col in multilayer_metrics.columns 
                          if col not in ['future_return', 'crisis']]
        
        # Filtrar características con demasiados NaN
        X = multilayer_metrics[feature_columns]
        X = X.loc[:, X.isnull().mean() < 0.5]  # Mantener columnas con menos de 50% NaN
        X = X.fillna(X.mean())
        
        y = multilayer_metrics['crisis']
        
        print(f" Características utilizadas: {len(X.columns)}")
        
        # Verificar que tenemos datos suficientes
        if len(X.columns) == 0:
            print(" No hay características válidas para entrenar")
            exit()
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Split temporal
        split_point = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled.iloc[:split_point], X_scaled.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        print(f" Entrenamiento: {len(X_train)} observaciones")
        print(f" Prueba: {len(X_test)} observaciones")
        
        # Verificar que tenemos casos de crisis en train y test
        if y_train.sum() == 0 or y_test.sum() == 0:
            print("  No hay suficientes casos de crisis para entrenamiento")
            print(f"Crisis en entrenamiento: {y_train.sum()}")
            print(f"Crisis en prueba: {y_test.sum()}")
        
        # Entrenar modelos
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n Entrenando {name}...")
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                auc_score = roc_auc_score(y_test, y_prob)
                
                results[name] = {
                    'model': model,
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'feature_importance': pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                }
                
                print(f" {name} - AUC: {auc_score:.3f}")
                print(classification_report(y_test, y_pred, target_names=['Normal', 'Crisis']))
                
            except Exception as e:
                print(f" Error entrenando {name}: {e}")
        
        # === PASO 7: ANALIZAR RESULTADOS ===
        if results:
            print("\n ANÁLISIS DE RESULTADOS")
            print("=" * 40)
            
            # Encontrar el mejor modelo
            best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
            best_model = results[best_model_name]
            
            print(f" Mejor modelo: {best_model_name}")
            print(f" AUC Score: {best_model['auc']:.3f}")
            
            # Obtener importancia de características
            feature_importance = best_model['feature_importance']
            
            print(f"\n Top 10 características más importantes:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
            
            # === ANÁLISIS POR CAPAS ===
            print(f"\n  ANÁLISIS POR CAPAS DE RED")
            print("=" * 40)
            
            # Analizar importancia por capa
            layer_importance = {}
            for _, row in feature_importance.iterrows():
                feature_name = row['feature']
                importance = row['importance']
                
                # Extraer nombre de la capa
                if '_' in feature_name:
                    layer = feature_name.split('_')[0]
                else:
                    layer = 'otros'
                
                if layer not in layer_importance:
                    layer_importance[layer] = 0
                layer_importance[layer] += importance
            
            print("Importancia acumulada por capa:")
            for layer, importance in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {layer:15s}: {importance:.4f}")
            
            # === VISUALIZACIONES ===
            print(f"\n📊 Generando visualizaciones...")
            
            # Gráfico 1: Importancia por capa
            plt.figure(figsize=(12, 6))
            layers = list(layer_importance.keys())
            importances = list(layer_importance.values())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
            bars = plt.bar(layers, importances, color=colors)
            
            plt.title('Importancia Acumulada por Capa de Red Multicapa', fontsize=16, fontweight='bold')
            plt.xlabel('Capa de Red', fontsize=12)
            plt.ylabel('Importancia Acumulada', fontsize=12)
            plt.xticks(rotation=45)
            
            # Agregar valores en las barras
            for bar, importance in zip(bars, importances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Gráfico 2: Top características
            plt.figure(figsize=(14, 8))
            top_features = feature_importance.head(15)
            
            # Crear colores por capa
            feature_colors = []
            unique_layers = list(set([f.split('_')[0] for f in top_features['feature'] if '_' in f]))
            color_map = dict(zip(unique_layers, plt.cm.Set3(np.linspace(0, 1, len(unique_layers)))))
            
            for feature in top_features['feature']:
                layer = feature.split('_')[0] if '_' in feature else 'otros'
                feature_colors.append(color_map.get(layer, 'gray'))
            
            bars = plt.barh(range(len(top_features)), top_features['importance'], color=feature_colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importancia', fontsize=12)
            plt.title('Top 15 Características Más Importantes', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Agregar valores
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{importance:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # === RESUMEN FINAL ===
            print(f"\n RESUMEN DEL ANÁLISIS MULTICAPA")
            print("=" * 50)
            print(f" Total de características utilizadas: {len(X.columns)}")
            print(f"  Capas de red analizadas: {len(layer_importance)}")
            print(f" Mejor modelo: {best_model_name} (AUC: {best_model['auc']:.3f})")
            print(f" Períodos de crisis detectados: {multilayer_metrics['crisis'].sum()}")
            
            if layer_importance:
                capa_mas_importante = max(layer_importance.items(), key=lambda x: x[1])
                print(f" Capa más predictiva: {capa_mas_importante[0]} ({capa_mas_importante[1]:.3f})")
            
            print(f"\n El análisis multicapa proporciona una visión más completa")
            print(f"   del sistema financiero que métodos de una sola capa.")
            
        else:
            print(" No se pudieron entrenar modelos")
    else:
        print("  Datos insuficientes para entrenamiento de modelos")
        print(f"   Crisis detectadas: {multilayer_metrics['crisis'].sum()}")
        print(f"   Observaciones totales: {len(multilayer_metrics)}")
else:
    print(f" Ticker objetivo {target_ticker} no encontrado en los datos")

print(f"\n Análisis completado exitosamente!")
