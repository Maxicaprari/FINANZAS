!pip install alphacast
from alphacast import Alphacast
alphacast = Alphacast("ak_MMjFBDyZDlANYLwmqRx7")
dataset = alphacast.datasets.dataset(42438)
file = dataset.download_data(format = "csv", startDate=None, endDate=None, filterVariables = [], filterEntities = {})


df = dataset.download_data(format = "pandas", startDate=None, endDate=None, filterVariables = [], filterEntities = {})

import pandas as pd 
df['Date'] = pd.to_datetime(df['Date'])


df = df[df['Date'] >= '2025-01-01'].reset_index(drop=True)



import numpy as np
import matplotlib.pyplot as plt


df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2025-01-01'].reset_index(drop=True)

ccl_hist = df['CCL promedio ADRs'].values
dates_hist = df['Date']


n_simulations = 1000
n_days = 300
volatility = np.std(np.diff(ccl_hist))
last_ccl = ccl_hist[-1]

simulations = np.zeros((n_days, n_simulations))
for i in range(n_simulations):
    shocks = np.random.normal(0, volatility, n_days)
    simulations[:, i] = last_ccl + np.cumsum(shocks)


band_start_date = pd.Timestamp('2025-04-11')
sim_start_date  = dates_hist.iloc[-1] + pd.Timedelta(days=1)
sim_end_date    = sim_start_date + pd.Timedelta(days=n_days-1)


initial_lower_band = 1000
initial_upper_band = 1400
monthly_rate = 0.01
daily_up   = (1 + monthly_rate) ** (1/30)   # banda superior
daily_down = (1 - monthly_rate) ** (1/30)   # banda inferior



band_dates = pd.date_range(band_start_date, sim_end_date, freq='D')
n_band_days = len(band_dates)


lower_band_full = np.empty(n_band_days)
upper_band_full = np.empty(n_band_days)

lower = initial_lower_band
upper = initial_upper_band
for i in range(n_band_days):
    lower_band_full[i] = lower
    upper_band_full[i] = upper
    lower *= daily_down
    upper *= daily_up


offset = (sim_start_date - band_start_date).days          # cuántos días después del 11-Abr arranca la simulación
lower_band = lower_band_full[offset : offset + n_days]    # longitud n_days
upper_band = upper_band_full[offset : offset + n_days]

upper_exit = []
lower_exit = []

for i in range(n_simulations):
    for j in range(n_days):
        if simulations[j, i] > upper_band[j]:
            upper_exit.append((j, simulations[j, i]))
        elif simulations[j, i] < lower_band[j]:
            lower_exit.append((j, simulations[j, i]))


dates_sim = pd.date_range(start=dates_hist.iloc[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')

plt.figure(figsize=(14,7))
plt.plot(dates_hist, ccl_hist, color='black', label='CCL')
plt.plot(band_dates, upper_band_full, 'r--', label='Banda Superior')
plt.plot(band_dates, lower_band_full, 'g--', label='Banda Inferior')

for i in range(0, n_simulations, 20):
    plt.plot(dates_sim, simulations[:, i], color='blue', alpha=0.03)

if upper_exit:
    x_upper_exit, y_upper_exit = zip(*[(dates_sim[i], val) for i, val in upper_exit])
    plt.scatter(x_upper_exit, y_upper_exit, color='red', s=10, label='Salida Banda Superior')

if lower_exit:
    x_lower_exit, y_lower_exit = zip(*[(dates_sim[i], val) for i, val in lower_exit])
    plt.scatter(x_lower_exit, y_lower_exit, color='green', s=10, label='Salida Banda Inferior')

plt.axvline(x=dates_hist.iloc[-1], color='grey', linestyle='--', label='Inicio Simulación')
plt.title('Monte Carlo CCL con Bandas Móviles')
plt.xlabel('Fecha')
plt.ylabel('CCL (ARS/USD)')
plt.legend()
plt.grid(True)
plt.show()




total_exits = len(upper_exit) + len(lower_exit)
prob_exit = total_exits / (n_simulations * n_days)
print(f"\n--- Análisis estadístico de la simulación ---")
print(f"Probabilidad de salir de la banda en algún momento: {prob_exit*100:.2f}%")


exit_days = []          # primer día de salida (si la hay) por simulación
exit_side = []          # 'up' / 'down' para cada simulación
for i in range(n_simulations):
    first_day = None
    first_side = None
    for j in range(n_days):
        if not np.isnan(upper_band[j]):
            if simulations[j, i] > upper_band[j]:
                first_day = j; first_side = 'up'; break
            elif simulations[j, i] < lower_band[j]:
                first_day = j; first_side = 'down'; break
    if first_day is not None:
        exit_days.append(first_day)
        exit_side.append(first_side)


upper_exit_count = exit_side.count('up')
lower_exit_count = exit_side.count('down')
print(f"• Salidas por arriba : {upper_exit_count}")
print(f"• Salidas por abajo  : {lower_exit_count}")
print(f"• Prob. de no salir  : {(1 - len(exit_days)/n_simulations)*100:.2f}%")

if exit_days:                                 # stats de tiempo de salida
    print(f"Tiempo promedio hasta la 1ª salida   : {np.mean(exit_days):.1f} días")
    print(f"Mediana                               : {np.median(exit_days):.1f} días")
    print(f"Percentil 25-75 (RIC)                : {np.percentile(exit_days,25):.1f} – "
          f"{np.percentile(exit_days,75):.1f} días")
else:
    print("Ninguna simulación salió de la banda.")


final_ccls = simulations[-1, :]
mean_final = np.mean(final_ccls)
std_final  = np.std(final_ccls)
print(f"\n--- Estadística al día {n_days} ---")
print(f"CCL medio            : {mean_final:.2f} ARS/USD")
print(f"Desvío estándar      : {std_final:.2f} ARS/USD")

p10, p50, p90 = np.percentile(final_ccls, [10, 50, 90])
print(f"P10 (optimista)      : {p10:.2f}  |  Mediana: {p50:.2f}  |  P90 (pesimista): {p90:.2f}")


VaR_95  = np.percentile(final_ccls, 5)           # 5 % de cola inferior
CVaR_95 = final_ccls[final_ccls <= VaR_95].mean()
print(f"VaR 95 %             : {VaR_95:.2f} ARS/USD")
print(f"CVaR 95 %            : {CVaR_95:.2f} ARS/USD")


years          = n_days / 252      # 252 días hábiles ~ 1 año
ann_return     = (mean_final / last_ccl)**(1/years) - 1
ann_vol        = (std_final / mean_final) * np.sqrt(252)
print(f"Retorno anualizado   : {ann_return*100:.2f}%")
print(f"Volatilidad anualiz. : {ann_vol*100:.2f}%")


max_dd = []
for i in range(n_simulations):
    series = simulations[:, i]
    peak   = series[0]
    dd_sim = 0
    for price in series:
        peak = max(peak, price)
        dd_sim = max(dd_sim, (peak - price) / peak)
    max_dd.append(dd_sim)
print(f"Drawdown máx. prom.  : {np.mean(max_dd)*100:.2f}%")


import seaborn as sns

plt.figure(figsize=(10,5))
sns.histplot(final_ccls, kde=True, bins=40, color="steelblue")
plt.axvline(VaR_95, color='red', linestyle='--', label=f'VaR 95% = {VaR_95:.0f}')
plt.title('Distribución del CCL al final de la simulación')
plt.xlabel('CCL (ARS/USD)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
