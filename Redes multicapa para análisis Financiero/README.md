### Anáisis de Crisis Financieras mediante Redes Multicapa Supervisadas
## Máximo Caprari


El objetivo principal de este trabajo es desarrollar una metodología para la predicción de crisis financieras. Para ello, se propone el análisis de redes dinámicas construidas a partir de las correlaciones entre índices bursátiles globales. La hipótesis fundamental es que los períodos de inestabilidad y las crisis sistémicas se manifiestan como cambios estructurales detectables en la topología de estas redes. El fin último es utilizar las métricas derivadas de la red como características para entrenar modelos de aprendizaje supervisado capaces de generar alertas tempranas.


### MST_global.py

Este código implementa un análisis de redes financieras para estudiar las correlaciones entre diferentes índices bursátiles globales durante el período de 2020 a 2023
Primero descarga datos históricos de precios usando la librería yfinance para obtener información de múltiples tickers o símbolos financieros en el rango de fechas especificado Los datos se descargan con ajuste automático de precios para considerar dividendos y splits
Luego construye un DataFrame limpio extrayendo únicamente los precios de cierre ajustados de cada ticker verificando que cada símbolo esté presente en los datos descargados para evitar errores
El siguiente paso consiste en calcular los retornos logarítmicos de cada activo financiero usando la fórmula del logaritmo natural del precio actual dividido por el precio del período anterior Esta métrica es ampliamente utilizada en finanzas porque tiene propiedades matemáticas convenientes y aproxima bien los retornos porcentuales para cambios pequeños
Con los retornos logarítmicos calcula la matriz de correlación que muestra cómo se mueven los diferentes índices en relación unos con otros Los valores cercanos a 1 indican movimientos muy similares mientras que valores cercanos a -1 indican movimientos opuestos
Una innovación interesante del código es convertir las correlaciones en distancias usando la fórmula de distancia euclidiana en el espacio de correlaciones Esto permite interpretar correlaciones altas como distancias cortas entre activos financieros
Después construye un grafo completo usando NetworkX donde cada ticker es un nodo y cada par de nodos está conectado por una arista cuyo peso representa la distancia calculada anteriormente
El elemento central del análisis es la creación de un Minimum Spanning Tree o árbol de expansión mínima que encuentra la red de conexiones más eficiente que conecta todos los índices financieros minimizando la suma total de distancias Este MST revela la estructura subyacente de correlaciones en los mercados globales
Finalmente visualiza el árbol resultante usando un layout de resorte que posiciona los nodos de manera estéticamente agradable Los nodos representan los diferentes índices y las aristas muestran las conexiones más importantes con sus respectivos pesos de distancia
Esta metodología es útil para entender cómo se propagan las crisis financieras identificar grupos de mercados que se mueven juntos y detectar los índices que actúan como conectores principales en el sistema financiero global
