# Descripción del Proceso Analítico

## 1. Carga y preprocesamiento de datos  

El script parte de dos dataframes: df con variables explicativas y df_emae con el EMAE empalmado. Se convierten las columnas de fecha (fecha en df y Fecha en df_emae) a formato datetime, se fijan como índice y se ordenan en forma cronológica. Se eliminan columnas derivadas o redundantes (variaciones interanuales, índices de tendencia, etc.) según una lista de variables prohibidas. Se calcula la variación interanual del EMAE (y_series) y se alinean ambas bases sobre el mismo índice temporal.

## 2. Selección de variables explicativas  

Se seleccionan solo las columnas numéricas de df y se imputan valores faltantes con la mediana. Se entrena un RandomForestRegressor para explicar el crecimiento interanual del EMAE y se extraen las importancias de cada variable. A partir de estas importancias se eligen las 20 variables más relevantes, que serán la base para el resto del análisis.

## 3. Construcción de ventanas temporales y PCA  

Con las variables seleccionadas se construyen ventanas móviles de 12 meses, que se aplanan en vectores y forman la matriz Z. Se estandariza Z y se aplica PCA para reducir la dimensión, eligiendo de forma automática el número de componentes necesario para explicar entre 70 % y 85 % de la varianza. El resultado son embeddings de menor dimensión que condensan la información conjunta de las series y del historial reciente.

## 4. Clustering espectral con estructura temporal  

Sobre los embeddings se define una matriz de afinidad que combina similitudes en el espacio de características y en el tiempo. La función spectral_clustering_temporal ajusta de manera automática los parámetros de escala espacial y temporal y estima el número de clústeres usando el criterio del eigengap. Con esta matriz de afinidad se aplica SpectralClustering y se obtienen etiquetas de régimen para cada fecha (regímenes iniciales).

## 5. Validación con modelo de cambio de régimen de Markov  

La serie de crecimiento del EMAE se alinea con las fechas de las ventanas y se estima un modelo MarkovRegression con varianza cambiante entre regímenes. Si la estimación converge, se obtienen probabilidades suavizadas de cada régimen y una clasificación alternativa de regímenes de Markov, que sirve como contraste del clustering espectral.

## 6. Renumeración e interpretación de regímenes  

Los regímenes detectados se renumeran según el orden temporal de aparición para facilitar la lectura. Para cada régimen se calculan duración, media y volatilidad del crecimiento, mínimos y máximos, y fechas de inicio y fin. Toda esta información se guarda en regime_interpretations, junto con un color preasignado para facilitar la visualización en gráficos.

## 7. Entradas, salidas y dependencias  

Entradas principales: df (variables explicativas con columna fecha) y df_emae (serie emae_empalmado con columna Fecha). Salidas principales: series de regímenes (regimes, regime_series), matriz de afinidad y parámetros del clustering (spectral_affinity, spectral_params), clasificación de Markov (markov_regimes, si está disponible) y el diccionario regime_interpretations. Dependencias: pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib y collections.

