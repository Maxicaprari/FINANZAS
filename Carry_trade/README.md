# Bonos CER – Carry, Breakeven & Estrategia de Salida  
**Jupyter Notebook · Python 3**

Este repositorio contiene un notebook que calcula métricas de _carry_ para bonos CER argentinos, grafica el breakeven MEP y simula una estrategia de salida anticipada basada en compresión de tasa.

---

## 1. ¿Qué hace el notebook?

### ✅ Define la cartera
Incluye una lista de 25 bonos CER con:
- Ticker
- Fecha de vencimiento
- Payoff (valor técnico estimado al maturity)

### ✅ Descarga precios en vivo
Consulta tres endpoints de [data912](https://data912.com):
- `/live/mep`: cotización MEP (mediana)
- `/live/arg_notes` y `/live/arg_bonds`: último precio, bid y ask de cada bono

### ✅ Construye el DataFrame `carry`
Cruza los precios con la lista de bonos y calcula:
- Días al vencimiento
- Tasa Nominal Anual (TNA)
- Tasa Efectiva Anual (TEA)
- Tasa Efectiva Mensual (TEM)
- _Carry_ en distintos escenarios de tipo de cambio (USD a ARS 1000–1400)
- _Carry_ “worst” suponiendo dólar subiendo 1 % mensual
- `MEP_BREAKEVEN`: precio USD al que el bono iguala la banda superior

### ✅ Gráfico en modo oscuro
- Banda superior (`finish_worst`) en naranja
- Breakeven en blanco
- Anotaciones por ticker

### ✅ Simulación de salida anticipada (`carry_exit`)
Modela la rentabilidad en ARS saliendo el **15-Oct-2025**, con una **TEM objetivo del 1 %**
- Muestra el yield directo y su TEA equivalente

---

## 2. Requisitos

- Python ≥ 3.9  
- Conexión a internet para consultar la API de [data912.com](https://data912.com)

---

## 3. Uso rápido

1. Cloná este repo.
2. Abrí el notebook en Jupyter o VS Code.
3. Ejecutá la primera celda (incluye `%%time` para medir performance).
4. Se mostrarán:
   - Dos tablas formateadas: `carry` y `carry_exit`
   - Un gráfico en modo oscuro

---

## 4. Descripción de columnas clave

| Columna           | Significado                                                             |
|-------------------|-------------------------------------------------------------------------|
| `bond_price`      | Último precio de mercado                                                |
| `payoff`          | Valor técnico estimado al vencimiento                                   |
| `days_to_exp`     | Días calendario hasta el maturity                                       |
| `tna`, `tea`, `tem` | Tasas implícitas calculadas sobre `bond_price`                         |
| `carry_xxxx`      | _Carry_ si el USD vale `xxxx` ARS al vencimiento                        |
| `carry_worst`     | _Carry_ bajo escenario “USD +1 % mensual”                               |
| `finish_worst`    | Proyección de USD “worst” a la fecha de vencimiento                    |
| `MEP_BREAKEVEN`   | Precio USD al que el bono toca la banda superior                        |
| `ars_direct_yield`| Rendimiento directo ARS en la salida anticipada                        |
| `ars_tea`         | TEA equivalente del rendimiento ARS en la salida anticipada            |

---

## 5. Parámetros editables

| Variable         | Default           | Descripción                                                    |
|------------------|-------------------|----------------------------------------------------------------|
| `price_scenarios`| `[1000,1100,1200,1300,1400]` | Escenarios de USD para calcular carry                  |
| `CPI_EST`        | `0.01`            | TEM objetivo al cierre de la estrategia en ARS                 |
| `EST_DATE`       | `2025-10-15`      | Fecha de salida para la simulación de estrategia               |

Podés modificar estos valores al inicio del notebook para testear distintos supuestos.

---

## 6. Estructura de archivos

(Ejemplo sugerido – completá según tu repo)

