!pip install tvscreener -q

import tvscreener as tvs
from tvscreener import StockScreener, StockField
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)

ss = StockScreener()

ss.select(
    StockField.NAME,
    StockField.PRICE,
    StockField.CHANGE_PERCENT,
    StockField.VOLUME,
    StockField.MARKET_CAPITALIZATION,
    StockField.RELATIVE_STRENGTH_INDEX_14,
    StockField.MACD_LEVEL_12_26,
    StockField.SIMPLE_MOVING_AVERAGE_50,
    StockField.SIMPLE_MOVING_AVERAGE_200,
    StockField.MONTH_HIGH_1,
    StockField.MONTH_LOW_1,
    StockField.MONTH_PERFORMANCE_3,
    StockField.WEEK_HIGH_52,
    StockField.WEEK_LOW_52,
    StockField.YEAR_BETA_1,
)

ss.set_range(0, 200)
df = ss.get()

top_gainers = df.nlargest(20, 'Change %')[['Name', 'Symbol', 'Change %', 'Price', 'Volume']].copy()
top_losers = df.nsmallest(20, 'Change %')[['Name', 'Symbol', 'Change %', 'Price', 'Volume']].copy()

oversold = df[df['Relative Strength Index (14)'] < 30].sort_values('Relative Strength Index (14)')
overbought = df[df['Relative Strength Index (14)'] > 70].sort_values('Relative Strength Index (14)', ascending=False)

golden = df[df['Simple Moving Average (50)'] > df['Simple Moving Average (200)']].copy()
death = df[df['Simple Moving Average (50)'] < df['Simple Moving Average (200)']].copy()

df['Dist_52W_High_%'] = ((df['Price'] - df['52 Week High']) / df['52 Week High'] * 100)
df['Dist_52W_Low_%'] = ((df['Price'] - df['52 Week Low']) / df['52 Week Low'] * 100)

near_highs = df[df['Dist_52W_High_%'] > -5].sort_values('Dist_52W_High_%', ascending=False)
near_lows = df[df['Dist_52W_Low_%'] < 5].sort_values('Dist_52W_Low_%')

high_beta = df.nlargest(15, '1-Year Beta')
low_beta = df.nsmallest(15, '1-Year Beta')

fig, axes = plt.subplots(3, 3, figsize=(20, 14))

ax = axes[0, 0]
top_g = df.nlargest(15, 'Change %')
colors = ['green' if x > 0 else 'red' for x in top_g['Change %']]
ax.barh(range(len(top_g)), top_g['Change %'], color=colors)
ax.set_yticks(range(len(top_g)))
ax.set_yticklabels(top_g['Name'], fontsize=8)
ax.set_xlabel('Cambio %')
ax.set_title('Top 15 Ganadores', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

ax = axes[0, 1]
top_l = df.nsmallest(15, 'Change %')
colors = ['green' if x > 0 else 'red' for x in top_l['Change %']]
ax.barh(range(len(top_l)), top_l['Change %'], color=colors)
ax.set_yticks(range(len(top_l)))
ax.set_yticklabels(top_l['Name'], fontsize=8)
ax.set_xlabel('Cambio %')
ax.set_title('Top 15 Perdedores', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

ax = axes[0, 2]
ax.hist(df['Change %'].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='0%')
ax.axvline(df['Change %'].mean(), color='orange', linestyle='-', linewidth=2, 
           label=f'Media: {df["Change %"].mean():.2f}%')
ax.set_xlabel('Cambio %')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Cambios', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
rsi_data = df['Relative Strength Index (14)'].dropna()
ax.hist(rsi_data, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(30, color='green', linestyle='--', linewidth=2, label='Sobreventa (30)')
ax.axvline(70, color='red', linestyle='--', linewidth=2, label='Sobrecompra (70)')
ax.axvline(rsi_data.mean(), color='orange', linestyle='-', linewidth=2, 
           label=f'Media: {rsi_data.mean():.1f}')
ax.set_xlabel('RSI')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución RSI', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
colors = ['green' if x > 0 else 'red' for x in df['Change %']]
ax.scatter(df['Relative Strength Index (14)'], df['Change %'], c=colors, alpha=0.5, s=30)
ax.axvline(30, color='green', linestyle='--', alpha=0.5, label='Sobreventa')
ax.axvline(70, color='red', linestyle='--', alpha=0.5, label='Sobrecompra')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('RSI')
ax.set_ylabel('Cambio %')
ax.set_title('RSI vs Cambio %', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 2]
colors = ['green' if x > 0 else 'red' for x in df['Change %']]
ax.scatter(df['Volume'], df['Change %'], c=colors, alpha=0.5, s=30)
ax.set_xlabel('Volumen')
ax.set_ylabel('Cambio %')
ax.set_title('Volumen vs Cambio %', fontweight='bold', fontsize=12)
ax.set_xscale('log')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3)

ax = axes[2, 0]
perf_data = df['3-Month Performance'].dropna()
ax.hist(perf_data, bins=40, color='darkgreen', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.axvline(perf_data.mean(), color='orange', linestyle='-', linewidth=2, 
           label=f'Media: {perf_data.mean():.1f}%')
ax.set_xlabel('Performance 3 Meses %')
ax.set_ylabel('Frecuencia')
ax.set_title('Performance 3 Meses', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 1]
beta_data = df['1-Year Beta'].dropna()
beta_data = beta_data[(beta_data > -0.5) & (beta_data < 2.5)]
ax.hist(beta_data, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(1, color='black', linestyle='--', linewidth=2, label='Beta = 1 (Mercado)')
ax.axvline(beta_data.mean(), color='blue', linestyle='-', linewidth=2, 
           label=f'Media: {beta_data.mean():.2f}')
ax.set_xlabel('Beta')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución Beta', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2, 2]
dist_data = df['Dist_52W_High_%'].dropna()
ax.hist(dist_data, bins=40, color='navy', alpha=0.7, edgecolor='black')
ax.axvline(-5, color='green', linestyle='--', linewidth=2, label='Cerca del máximo')
ax.axvline(dist_data.mean(), color='orange', linestyle='-', linewidth=2, 
           label=f'Media: {dist_data.mean():.1f}%')
ax.set_xlabel('Distancia de Máximo 52W %')
ax.set_ylabel('Frecuencia')
ax.set_title('Distancia de Máximos 52W', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_stocks.png', dpi=300, bbox_inches='tight')
plt.show()

df.to_csv('stocks_completo.csv', index=False)
top_gainers.to_csv('top_ganadores.csv', index=False)
top_losers.to_csv('top_perdedores.csv', index=False)

if len(oversold) > 0:
    oversold.to_csv('sobreventa_rsi.csv', index=False)

if len(overbought) > 0:
    overbought.to_csv('sobrecompra_rsi.csv', index=False)

if len(golden) > 0:
    golden.to_csv('golden_cross.csv', index=False)

if len(near_highs) > 0:
    near_highs.to_csv('cerca_maximos_52w.csv', index=False)

if len(near_lows) > 0:
    near_lows.to_csv('cerca_minimos_52w.csv', index=False)
