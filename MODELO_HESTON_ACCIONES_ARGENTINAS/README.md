# Calibración del Modelo de Heston en el Mercado Argentino

Este repositorio contiene la implementación y análisis de volatilidad estocástica en acciones argentinas durante el período 2020-2025 usando el modelo de Heston.

## Descripción

El trabajo calibra el modelo de Heston sobre 21 activos representativos del mercado local, incluyendo acciones líderes de diferentes sectores y el índice MERVAL. Se implementa una metodología mejorada de calibración usando evolución diferencial en lugar de optimizadores locales tradicionales, lo que permite capturar heterogeneidad real entre activos que antes quedaba oculta por problemas de convergencia.

## Contenido

- `codigo.py`: Script principal con calibración completa
- `graficos.py`: Generación de gráficos para MERVAL
- `latex/`: Documentación completa en LaTeX


## Resultados principales

La calibración mejorada revela:

- Velocidad de reversión entre 3.85 (AGRO) y 14.97 (BMA), con la mayoría de activos entre 12 y 15
- Correlación precio-volatilidad heterogénea: desde -0.95 (EDN, ALUA) hasta -0.08 (TGSU2)
- Tres grupos identificados según intensidad del efecto leverage
- Todos los activos cumplen la condición de Feller
- Las simulaciones Monte Carlo muestran convergencia de volatilidad desde niveles extremos (~115%) hacia equilibrios de largo plazo (~15%) en horizontes anuales

## Metodología

### Calibración

El proceso de calibración sigue estos pasos:

1. Descarga de precios históricos (5 años) desde Yahoo Finance
2. Estimación de varianza condicional usando GARCH(1,1) como proxy
3. Calibración de parámetros Heston por máxima verosimilitud con evolución diferencial
4. Validación de la condición de Feller para cada activo
5. Simulación de trayectorias futuras usando el esquema QE de Andersen (2008)

### Mejoras respecto a implementaciones estándar

- Uso de evolución diferencial en lugar de L-BFGS-B para evitar mínimos locales
- Bounds más amplios que permiten mayor variabilidad entre activos
- Penalización explícita por violación de la condición de Feller
- Validación exhaustiva de resultados con múltiples métricas

## Requisitos

```
python >= 3.8
numpy
pandas
scipy
matplotlib
yfinance
arch
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso

### Calibración completa de todos los activos

```python
python heston_calibration.py
```

Esto genera:
- Parámetros calibrados para cada activo
- Tablas LaTeX con resultados
- Gráficos de comparación GARCH vs Heston
- Simulaciones Monte Carlo

### Visualizaciones rápidas para MERVAL

```python
python heston_visualization.py
```

Genera tres gráficos optimizados para el índice MERVAL:
- Comparación de volatilidades GARCH vs Heston
- Trayectorias simuladas de precio
- Trayectorias simuladas de volatilidad

## Interpretación de parámetros

Para el caso del MERVAL:

- kappa = 14.88: Tiempo medio de reversión de ~17 días hábiles
- theta = 0.000091: Volatilidad de largo plazo de ~15%
- xi = 0.0504: Volatilidad de la varianza moderada
- rho = -0.49: Efecto leverage moderado
- V_0 = 0.005179: Volatilidad inicial extremadamente alta (~114%)

## Estructura del modelo

El modelo de Heston describe la evolución conjunta del precio y su varianza:

```
dS_t = r S_t dt + sqrt(V_t) S_t dW_1
dV_t = kappa(theta - V_t)dt + xi sqrt(V_t) dW_2
```

donde los procesos de Wiener están correlacionados con E[dW_1 dW_2] = rho dt.

### Condición de Feller

Para garantizar que la varianza no colapsa a cero, se requiere:

```
2 * kappa * theta > xi^2
```

Todos los activos analizados cumplen esta condición.

## Comparación con GARCH

El modelo GARCH captura bien shocks de corto plazo y clusters de volatilidad, pero carece de:
- Estructura de reversión a la media explícita
- Correlación precio-volatilidad (efecto leverage)
- Framework continuo para valoración de derivados

El modelo de Heston complementa GARCH al ofrecer una visión estructural de largo plazo adecuada para simulaciones y valoración.

## Limitaciones

El análisis tiene varias limitaciones importantes:

- Calibración en dos etapas (GARCH → Heston) introduce ruido de estimación
- Parámetros constantes en el tiempo, lo cual es restrictivo para el mercado argentino
- No incorpora saltos, frecuentes en mercados emergentes
- Sensibilidad alta a condiciones iniciales, especialmente V_0
- El nivel de equilibrio theta podría estar subestimado si el período de calibración incluye intervalos de baja liquidez artificial

## Extensiones futuras

- Calibración directa sobre precios de opciones cuando haya datos disponibles
- Implementación de modelos con saltos (Bates, Merton)
- Parámetros variables en el tiempo usando filtros de Kalman
- Análisis de sensibilidad y backtesting de predicciones
- Comparación con otros modelos de volatilidad estocástica (SABR, GARCH multivariado)

## Referencias

- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility. Review of Financial Studies, 6(2), 327-343.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance, 11(3), 1-42.
- Cox, J., Ingersoll, J., & Ross, S. (1985). A theory of the term structure of interest rates. Econometrica, 53(2), 385-407.

## Contexto

El período 2020-2025 representa uno de los intervalos de mayor turbulencia en la economía argentina reciente, incluyendo la pandemia de COVID-19, aceleración inflacionaria por encima del 100% anual, crisis cambiarias recurrentes, elecciones presidenciales con alta polarización y cambios significativos de política económica. Este contexto explica tanto los valores elevados de varianza inicial observados al final del período como la heterogeneidad en los parámetros de leverage entre sectores.


## Autor

Máximo Caprari


