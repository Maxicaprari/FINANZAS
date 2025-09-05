# Análisis multivariado de activos financieros

Este proyecto implementa un pipeline completo para analizar activos financieros mediante estadística multivariada, métricas de red y clusterización.  
El objetivo es identificar patrones comunes entre acciones, índices, commodities, FX y tasas a través de métricas cuantitativas y visualizaciones.

## Funcionalidades principales

- Descarga de datos
  - Con yfinance
  - O mediante API personalizada (con token opcional)

- Transformaciones de datos
  - Cálculo de retornos logarítmicos
  - Limpieza de series y normalización

- Métricas en ventanas móviles
  - Media, volatilidad y Sharpe anualizados
  - Retornos acumulados
  - Métricas de red financiera (correlaciones, spanning tree, clustering coeficiente)

- Análisis multivariado
  - PCA (Análisis de Componentes Principales)
  - Clusterización KMeans en espacio PCA
  - Clusterización jerárquica con dendrograma

- Visualizaciones
  - Scree plot (varianza explicada)
  - Biplot
  - Heatmap de correlaciones
  - Proyección temporal en espacio PCA
  - Clusterización con centroides y trayectorias
  - Dendrograma jerárquico

## Dependencias

El proyecto utiliza las siguientes librerías de Python:

```bash
pip install numpy pandas matplotlib seaborn networkx scikit-learn yfinance scipy




## Ejemplo de resultados

- Heatmap de correlaciones
- Clusterización PCA + KMeans
- Dendrograma jerárquico

## Conclusiones

- Los activos se agrupan en bloques diferenciados:
  - FX emergente y commodities
  - Acciones argentinas
  - Riesgo global (índices, tasas largas, petróleo)

- PCA y KMeans permiten reducir dimensionalidad y visualizar relaciones clave
- Las métricas de red aportan una visión estructural sobre las correlaciones de mercado

## Próximos pasos

- Incluir modelos de series temporales (VAR, GARCH, LSTM)
- Métricas de red dinámicas avanzadas
- Dashboard interactivo con Streamlit o Dash

