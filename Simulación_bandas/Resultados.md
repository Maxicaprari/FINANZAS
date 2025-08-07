## Lectura cualitativa de los resultados y estrategia cambiaria

### Probabilidad de salida de banda

- El **63,4 %** de las trayectorias (100 % − 36,6 %) terminan vulnerando el corredor en algún momento.
- De esas rupturas, el **84 %** (537 sobre 634) son por arriba: la presión predominante es de **depreciación** (suba del CCL).
- La **probabilidad diaria instantánea** de escape (18,9 %) es bastante alta y sugiere que la banda resulta **angosta** respecto a la volatilidad histórica asumida.

### Timing de la ruptura

- La mediana es de **61 días** con un rango intercuartílico de **27 a 124 días**: en la mitad de los casos el esquema se rompe antes de dos meses, y en el 75 % antes de cuatro.
- El promedio de **88 días** se ubica por encima de la mediana por la cola larga de rupturas tardías. Esto indica que **cuanto más resiste el corredor, menos probable es que se rompa más adelante**.

### Nivel del tipo de cambio al día 300

- Media ≈ **1.358** ARS/USD y mediana ≈ **1.355**, con una ligera asimetría positiva: P90 = 1.716 vs P10 = 994.
- El **VaR 95 %** (876) y el **CVaR 95 %** (746) indican que solo un 5 % de los escenarios terminan por debajo de 876 ARS/USD: la **cola apreciatoria es corta**.
- La parte alta (P90) queda un 26 % arriba de la mediana, y la parte baja (P10) un 27 % por debajo: hay cierta simetría, pero **la cola derecha es algo más pesada** (rompimientos por arriba).

### Retorno y riesgo anualizados

- El **retorno anualizado** es de apenas **0,8 %**, con una **volatilidad anualizada** de **335 %**: la relación retorno/riesgo es extremadamente baja.
- El proceso simulado es esencialmente un **paseo aleatorio sin drift**, con una volatilidad muy superior a la de activos financieros típicos.
- Esto indica que **el ancho de banda no encorseta lo suficiente la incertidumbre cambiaria**.

### Drawdown máximo promedio

- En promedio, cada trayectoria sufre una caída de aproximadamente **23 %** respecto a su pico: la banda no evita el riesgo de **apreciaciones bruscas**.

## Síntesis estratégica

### Credibilidad del corredor

- Dos de cada tres trayectorias rompen la banda, mayormente por arriba: el mercado **percibe que el crawling-peg no contiene la presión devaluatoria**.

### Ajustes posibles al régimen de bandas

- **Aumentar la pendiente del crawl** (acelerar el % diario) para trasladar parte de la depreciación esperada dentro del rango.
- **Ensachar las bandas iniciales** o hacerlas dinámicas (basadas en la volatilidad implícita) para reducir la frecuencia de escapes.
- **Complementar con intervenciones puntuales** o instrumentos que absorban shocks (futuros, bonos dollar-linked, encajes sobre posiciones en USD).

### Gestión del riesgo

- El **VaR/CVaR reflejan bajo riesgo de apreciación fuerte**: si el foco es evitar pérdida de competitividad, el esquema es relativamente seguro.
- El **sesgo hacia rupturas por arriba** indica que la reputación del régimen está más expuesta al lado de la depreciación.
- Es clave monitorear **reservas internacionales** y el **ritmo de intervención** cambiaria.

### Consideraciones de modelado

- El modelo se basa en un proceso **sin drift y con varianza constante**.
- Incorporar:
  - Drift macroeconómico,
  - Mean-reversion,
  - Choques discretos (event-risk),
  
  podría **refinar la probabilidad de ruptura y la forma de las colas de la distribución**.

### Conclusión

Con los parámetros actuales, el régimen de bandas ofrece una **ventana de credibilidad de entre uno y dos meses**. Para prolongarla se sugieren tres alternativas:

1. Mover más rápido el techo del corredor cambiario.
2. Dar mayor ancho inicial a las bandas.
3. Reducir la volatilidad efectiva mediante intervenciones o regulaciones.
