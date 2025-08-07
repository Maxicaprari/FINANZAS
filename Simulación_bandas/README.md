# Simulación Monte Carlo del CCL con Bandas Móviles

Este proyecto implementa una simulación de Monte Carlo para proyectar la evolución del **Contado con Liquidación (CCL)** a partir de datos históricos obtenidos desde **Alphacast**, incorporando un análisis de riesgo basado en **bandas móviles de intervención** y métricas estadísticas financieras.

## Requisitos

- Python 3.8 o superior
- Paquetes:
  - `alphacast`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

### Instalación

```bash
pip install alphacast pandas numpy matplotlib seaborn

## Descripción del código

### Extracción de datos históricos

Se conecta a la API de Alphacast para descargar la serie de tiempo del CCL promedio (dataset ID: 42438).

Se filtran los datos a partir del 1 de enero de 2025.

### Parámetros de simulación

Se calcula la volatilidad diaria del CCL a partir de diferencias logarítmicas.

Se simulan 1000 trayectorias de 300 días utilizando movimientos brownianos acumulativos.

### Bandas móviles de intervención

Inician el 11 de abril de 2025.

Valores iniciales:
- Banda inferior: 1000 ARS/USD
- Banda superior: 1400 ARS/USD

Ajuste diario equivalente a una tasa del 1 % mensual en términos de TEA.

Las bandas se ajustan exponencialmente día a día.

### Detección de salidas

Se detecta si cada simulación supera la banda superior o cae por debajo de la banda inferior.

Se registra el momento de salida y la magnitud para cada trayectoria.

### Visualizaciones

- Gráfico de evolución histórica del CCL.
- Gráfico de bandas móviles.
- Trayectorias simuladas.
- Puntos de salida de banda.
- Histograma de valores finales del CCL al día 300 con VaR y CVaR al 95 %.

### Métricas estadísticas

- Probabilidad de salida de banda.
- Tiempo promedio hasta la primera salida.
- Cantidad de salidas superiores e inferiores.
- Estadísticas del valor final del CCL:
  - Media, desviación estándar, percentiles 10, 50 y 90.
- Medidas de riesgo:
  - VaR (Value-at-Risk) al 95 %
  - CVaR (Conditional VaR) al 95 %
- Indicadores financieros:
  - Retorno anualizado
  - Volatilidad anualizada
  - Drawdown máximo promedio

## Interpretación

Este modelo permite estimar el riesgo de que el CCL cruce bandas de intervención simuladas bajo un escenario de caminata aleatoria con volatilidad histórica constante. La herramienta es útil para análisis de stress financiero, evaluación de riesgo cambiario y simulación de escenarios extremos.
