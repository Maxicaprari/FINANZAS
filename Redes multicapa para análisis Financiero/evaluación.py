import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# === CÓDIGO PARA GENERAR GRÁFICA DE EVOLUCIÓN TEMPORAL DE PERFORMANCE ===

def evaluate_temporal_performance(multilayer_metrics, window_evaluation=50, step_eval=10):
    """
    Evalúa la performance del modelo en diferentes ventanas temporales
    """
    # Preparar datos
    feature_columns = [col for col in multilayer_metrics.columns 
                      if col not in ['future_return', 'crisis']]
    
    X = multilayer_metrics[feature_columns]
    X = X.loc[:, X.isnull().mean() < 0.5]
    X = X.fillna(X.mean())
    y = multilayer_metrics['crisis']
    
    # Escalar características
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    # Evaluar en ventanas temporales
    performance_results = []
    
    # Usar ventanas deslizantes para evaluación
    min_train_size = 60
    
    for end_idx in range(min_train_size + window_evaluation, len(X_scaled), step_eval):
        # Definir ventana de entrenamiento y prueba
        train_end = end_idx - window_evaluation
        train_start = max(0, train_end - min_train_size)
        
        X_train_temp = X_scaled.iloc[train_start:train_end]
        y_train_temp = y.iloc[train_start:train_end]
        X_test_temp = X_scaled.iloc[train_end:end_idx]
        y_test_temp = y.iloc[train_end:end_idx]
        
        # Solo evaluar si tenemos casos positivos y negativos en entrenamiento
        if y_train_temp.sum() > 0 and (len(y_train_temp) - y_train_temp.sum()) > 0:
            if len(y_test_temp) > 5:  # Mínimo de observaciones para test
                try:
                    # Entrenar modelo
                    from sklearn.ensemble import GradientBoostingClassifier
                    model_temp = GradientBoostingClassifier(n_estimators=50, random_state=42)
                    model_temp.fit(X_train_temp, y_train_temp)
                    
                    # Predecir
                    y_pred_temp = model_temp.predict(X_test_temp)
                    y_prob_temp = model_temp.predict_proba(X_test_temp)[:, 1]
                    
                    # Calcular métricas
                    auc = roc_auc_score(y_test_temp, y_prob_temp) if len(np.unique(y_test_temp)) > 1 else 0
                    accuracy = accuracy_score(y_test_temp, y_pred_temp)
                    
                    # Precision y recall (manejar casos sin positivos)
                    if y_test_temp.sum() > 0:
                        precision = precision_score(y_test_temp, y_pred_temp, zero_division=0)
                        recall = recall_score(y_test_temp, y_pred_temp, zero_division=0)
                    else:
                        precision = 0
                        recall = 0
                    
                    # F1 Score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    performance_results.append({
                        'date': X_scaled.index[end_idx - 1],
                        'auc': auc,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'crisis_in_period': y_test_temp.sum(),
                        'total_observations': len(y_test_temp)
                    })
                    
                except Exception as e:
                    print(f"Error en ventana {end_idx}: {e}")
                    continue
    
    return pd.DataFrame(performance_results)

# Ejecutar evaluación temporal
print("\n📈 Evaluando performance temporal...")
temporal_performance = evaluate_temporal_performance(multilayer_metrics)

if not temporal_performance.empty:
    print(f"✅ Performance evaluada en {len(temporal_performance)} ventanas temporales")
    
    # === CREAR GRÁFICA DE EVOLUCIÓN TEMPORAL ===
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Evolución Temporal de Métricas de Performance del Modelo Multicapa', 
                fontsize=16, fontweight='bold')
    
    # Configurar fechas para el eje x
    dates = temporal_performance['date']
    
    # Gráfica 1: AUC
    axes[0, 0].plot(dates, temporal_performance['auc'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Evolución del AUC Score', fontweight='bold')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Línea de referencia en 0.5
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random (0.5)')
    axes[0, 0].legend()
    
    # Gráfica 2: Accuracy
    axes[0, 1].plot(dates, temporal_performance['accuracy'], 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Evolución de la Exactitud (Accuracy)', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Gráfica 3: Precision y Recall
    axes[1, 0].plot(dates, temporal_performance['precision'], 'purple', linewidth=2, 
                   marker='^', markersize=4, label='Precision')
    axes[1, 0].plot(dates, temporal_performance['recall'], 'orange', linewidth=2, 
                   marker='v', markersize=4, label='Recall')
    axes[1, 0].set_title('Evolución de Precision y Recall', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    
    # Gráfica 4: F1-Score y Crisis detectadas
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # F1-Score
    line1 = ax4.plot(dates, temporal_performance['f1_score'], 'red', linewidth=2, 
                    marker='d', markersize=4, label='F1-Score')
    ax4.set_ylabel('F1-Score', color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.set_ylim(0, 1)
    
    # Crisis detectadas (barras)
    bars = ax4_twin.bar(dates, temporal_performance['crisis_in_period'], 
                       alpha=0.3, color='gray', width=20, label='Crisis Detectadas')
    ax4_twin.set_ylabel('Número de Crisis', color='gray')
    ax4_twin.tick_params(axis='y', labelcolor='gray')
    
    ax4.set_title('F1-Score y Crisis Detectadas por Período', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Combinar leyendas
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Configurar fechas en todos los subplots
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('Fecha')
    
    plt.tight_layout()
    plt.show()
    
    # === ESTADÍSTICAS RESUMEN ===
    print("\n📊 ESTADÍSTICAS DE PERFORMANCE TEMPORAL")
    print("=" * 50)
    print(f"AUC promedio: {temporal_performance['auc'].mean():.3f} ± {temporal_performance['auc'].std():.3f}")
    print(f"AUC máximo: {temporal_performance['auc'].max():.3f}")
    print(f"AUC mínimo: {temporal_performance['auc'].min():.3f}")
    print(f"Accuracy promedio: {temporal_performance['accuracy'].mean():.3f} ± {temporal_performance['accuracy'].std():.3f}")
    print(f"F1-Score promedio: {temporal_performance['f1_score'].mean():.3f} ± {temporal_performance['f1_score'].std():.3f}")
    print(f"Total de crisis detectadas: {temporal_performance['crisis_in_period'].sum()}")
    print(f"Períodos con crisis: {(temporal_performance['crisis_in_period'] > 0).sum()}")
    
    # === ANÁLISIS DE ESTABILIDAD ===
    print(f"\n🔍 ANÁLISIS DE ESTABILIDAD")
    print("=" * 30)
    
    # Calcular estabilidad del AUC
    auc_rolling_std = temporal_performance['auc'].rolling(window=5).std()
    print(f"Desviación estándar móvil del AUC (ventana 5): {auc_rolling_std.mean():.3f}")
    
    # Identificar períodos de mejor performance
    high_performance_periods = temporal_performance[temporal_performance['auc'] > 0.7]
    if not high_performance_periods.empty:
        print(f"Períodos con AUC > 0.7: {len(high_performance_periods)}")
        print("Fechas de alta performance:")
        for date in high_performance_periods['date'].head(3):
            print(f"  - {date.strftime('%Y-%m-%d')}")
    
    # Correlación entre métricas
    corr_matrix = temporal_performance[['auc', 'accuracy', 'precision', 'recall', 'f1_score']].corr()
    print(f"\nCorrelación AUC-Accuracy: {corr_matrix.loc['auc', 'accuracy']:.3f}")
    print(f"Correlación AUC-F1: {corr_matrix.loc['auc', 'f1_score']:.3f}")

else:
    print("❌ No se pudo evaluar performance temporal")

# === GRÁFICA ADICIONAL: DISTRIBUCIÓN DE PERFORMANCE ===
if not temporal_performance.empty:
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Distribución de AUC
    plt.subplot(1, 2, 1)
    plt.hist(temporal_performance['auc'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(temporal_performance['auc'].mean(), color='red', linestyle='--', 
               label=f'Media: {temporal_performance["auc"].mean():.3f}')
    plt.axvline(0.5, color='orange', linestyle='--', label='Random: 0.5')
    plt.xlabel('AUC Score')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de AUC Score a lo Largo del Tiempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot de métricas
    plt.subplot(1, 2, 2)
    metrics_data = [
        temporal_performance['auc'],
        temporal_performance['accuracy'], 
        temporal_performance['precision'],
        temporal_performance['recall'],
        temporal_performance['f1_score']
    ]
    labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    box_plot = plt.boxplot(metrics_data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Distribución de Métricas de Performance')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
