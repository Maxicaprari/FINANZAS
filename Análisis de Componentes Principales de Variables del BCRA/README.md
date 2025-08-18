# Análisis de Componentes Principales de Variables del BCRA

Este script realiza un análisis de componentes principales (PCA) sobre datos económicos del Banco Central de la República Argentina (BCRA). A continuación, se detalla cada paso del proceso:

## Importación de Librerías

Se importan las librerías necesarias para el análisis: `requests` para realizar solicitudes HTTP, `pandas` para manipulación de datos, `sklearn` para el PCA y escalado, y `matplotlib` y `seaborn` para visualización.

## Definición de Variables

Se definen los IDs de las variables de interés y sus descripciones:

- **Reservas Internacionales**
- **Tipo de Cambio Minorista y Mayorista**
- **Base Monetaria**
- **Circulante en poder del público**
- **Depósitos y Crédito al sector privado en pesos**

## Descarga de Datos

Se descargan las series de datos desde la API del BCRA para cada variable de interés. Los datos se almacenan en un diccionario `series_dict` y se convierten en un DataFrame de pandas.

## Preprocesamiento de Datos

Se combinan las series en un solo DataFrame, eliminando las filas y columnas con valores nulos. Luego, los datos se escalan utilizando `StandardScaler` para normalizar las variables.

## Análisis de Componentes Principales (PCA)

Se aplica PCA a los datos escalados para reducir la dimensionalidad. Se calcula la varianza explicada por cada componente principal y se visualiza en un Scree Plot.

## Visualización de Resultados

- **Biplot:** Muestra las observaciones y las cargas de las variables en los dos primeros componentes principales.
- **Matriz de Correlación:** Visualiza las correlaciones entre las variables.
- **Series Temporales:** Se grafican las series originales y su promedio móvil de 30 días.
- **Proyección Temporal:** Muestra cómo las observaciones se distribuyen en el tiempo en el espacio PCA.

## Clusterización

Se aplica KMeans para agrupar las series en el espacio de los componentes principales. Los resultados se visualizan en un gráfico de dispersión.

## Exportación de Datos

El DataFrame procesado se puede guardar en un archivo CSV para su posterior análisis.

Este análisis proporciona una visión integral de cómo las variables económicas del BCRA interactúan y evolucionan con el tiempo.
