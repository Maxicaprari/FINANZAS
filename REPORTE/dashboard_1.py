!pip install tvscreener matplotlib -q

from tvscreener import StockScreener, StockField
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    import subprocess
    import platform

ss = StockScreener()

ss.select(
    StockField.NAME,
    StockField.PRICE,
    StockField.CHANGE_PERCENT,
    StockField.VOLUME,
    StockField.SECTOR,
    StockField.INDUSTRY,
)

ss.set_range(0, 200)
df = ss.get()

top_gainers = df.nlargest(20, 'Change %')[['Name', 'Sector', 'Industry', 'Change %', 'Price', 'Volume']].copy()
top_losers = df.nsmallest(20, 'Change %')[['Name', 'Sector', 'Industry', 'Change %', 'Price', 'Volume']].copy()

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

fig_tops, axes_tops = plt.subplots(2, 2, figsize=(16, 10))

sectores_unicos = pd.concat([top_gainers['Sector'], top_losers['Sector']]).unique()
colores_sectores = plt.cm.Blues(np.linspace(0.35, 0.85, len(sectores_unicos)))
mapa_colores = dict(zip(sectores_unicos, colores_sectores))

ax = axes_tops[0, 0]
colores_gainers = [mapa_colores.get(sector, 'gray') for sector in top_gainers['Sector']]
bars = ax.barh(range(len(top_gainers)), top_gainers['Change %'], color=colores_gainers)
ax.set_yticks(range(len(top_gainers)))
ax.set_yticklabels([f"{row['Name']} ({row['Sector']})" for _, row in top_gainers.iterrows()], fontsize=8)
ax.set_xlabel('Cambio %', fontsize=10)
ax.set_title('Top 20 Ganadores por Sector', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

ax = axes_tops[0, 1]
colores_losers = [mapa_colores.get(sector, 'gray') for sector in top_losers['Sector']]
bars = ax.barh(range(len(top_losers)), top_losers['Change %'], color=colores_losers)
ax.set_yticks(range(len(top_losers)))
ax.set_yticklabels([f"{row['Name']} ({row['Sector']})" for _, row in top_losers.iterrows()], fontsize=8)
ax.set_xlabel('Cambio %', fontsize=10)
ax.set_title('Top 20 Perdedores por Sector', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

ax = axes_tops[1, 0]
ganadores_por_sector = top_gainers['Sector'].value_counts()
perdedores_por_sector = top_losers['Sector'].value_counts()
sectores_combinados = ganadores_por_sector.index.union(perdedores_por_sector.index)
x = np.arange(len(sectores_combinados))
width = 0.35
ganadores_counts = [ganadores_por_sector.get(s, 0) for s in sectores_combinados]
perdedores_counts = [perdedores_por_sector.get(s, 0) for s in sectores_combinados]
ax.bar(x - width/2, ganadores_counts, width, label='Ganadores', color='#6ba368', alpha=0.8)
ax.bar(x + width/2, perdedores_counts, width, label='Perdedores', color='#c96b6b', alpha=0.8)
ax.set_xlabel('Sector', fontsize=10)
ax.set_ylabel('Cantidad', fontsize=10)
ax.set_title('Distribución de Ganadores vs Perdedores por Sector', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(sectores_combinados, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes_tops[1, 1]
cambio_promedio_ganadores = top_gainers.groupby('Sector')['Change %'].mean().sort_values(ascending=False)
cambio_promedio_perdedores = top_losers.groupby('Sector')['Change %'].mean().sort_values(ascending=True)
sectores_todos = list(cambio_promedio_ganadores.index) + list(cambio_promedio_perdedores.index)
x_pos = np.arange(len(sectores_todos))
valores = list(cambio_promedio_ganadores.values) + list(cambio_promedio_perdedores.values)
colores = ['#6ba368' if v > 0 else '#c96b6b' for v in valores]
ax.barh(x_pos, valores, color=colores, alpha=0.7)
ax.set_yticks(x_pos)
ax.set_yticklabels(sectores_todos, fontsize=8)
ax.set_xlabel('Cambio Promedio %', fontsize=10)
ax.set_title('Cambio Promedio por Sector (Top 20)', fontweight='bold', fontsize=12)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
img_graficos = fig_to_base64(fig_tops)
plt.close(fig_tops)

def df_to_html_table(df, title):
    html = f'<h3>{title}</h3>'
    html += '<div class="table-container">'
    html += df.to_html(classes='data-table', table_id=title.lower().replace(' ', '-'), escape=False, index=False)
    html += '</div>'
    return html

html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Análisis de Acciones</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f3f4f6;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(15,23,42,0.12);
            padding: 30px;
        }}

        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
        }}

        h1 {{
            color: #111827;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}

        .charts-section {{
            margin-bottom: 40px;
        }}

        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        .tables-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}

        .table-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        h3 {{
            color: #374151;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        .data-table th {{
            background: #e5e7eb;
            color: #111827;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}

        .data-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}

        .data-table tr:hover {{
            background: #f0f0f0;
        }}

        .data-table tr:nth-child(even) {{
            background: #fafafa;
        }}

        .positive {{
            color: #28a745;
            font-weight: 600;
        }}

        .negative {{
            color: #dc3545;
            font-weight: 600;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: #eef2ff;
            color: #111827;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(15,23,42,0.10);
        }}

        .stat-card h4 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}

        @media (max-width: 768px) {{
            .tables-section {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1> Dashboard de Análisis de Acciones</h1>
            <p class="timestamp">Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <h4>Total Acciones Analizadas</h4>
                <div class="value">{len(df)}</div>
            </div>
            <div class="stat-card">
                <h4>Ganadores</h4>
                <div class="value">{len(top_gainers)}</div>
            </div>
            <div class="stat-card">
                <h4>Perdedores</h4>
                <div class="value">{len(top_losers)}</div>
            </div>
            <div class="stat-card">
                <h4>Cambio Promedio</h4>
                <div class="value">{df['Change %'].mean():.2f}%</div>
            </div>
        </div>

        <div class="charts-section">
            <div class="chart-container">
                <h3>Análisis Visual por Sector</h3>
                <img src="data:image/png;base64,{img_graficos}" alt="Gráficos de Análisis">
            </div>
        </div>

        <div class="tables-section">
            {df_to_html_table(top_gainers, 'Top 20 Ganadores')}
            {df_to_html_table(top_losers, 'Top 20 Perdedores')}
        </div>
    </div>
</body>
</html>
"""

archivo_html = 'dashboard_stocks.html'
with open(archivo_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(" Dashboard HTML generado exitosamente")
print("\n Archivos creados:")
print(f"  {archivo_html} - Dashboard completo de análisis de acciones")

if IN_COLAB:
    print("\n Descargando archivo...")
    files.download(archivo_html)
    print("✓ Archivo descargado. Ábrelo en tu navegador.")
else:
    ruta_completa = os.path.abspath(archivo_html)
    carpeta = os.path.dirname(ruta_completa)
    print(f"\n Archivo guardado en: {ruta_completa}")

    def abrir_carpeta(ruta):
        sistema = platform.system()
        try:
            if sistema == 'Windows':
                os.startfile(ruta)
            elif sistema == 'Darwin':
                subprocess.run(['open', ruta])
            else:
                subprocess.run(['xdg-open', ruta])
        except:
            pass

    print("\n Abriendo carpeta con los archivos...")
    abrir_carpeta(carpeta)
    print(" Carpeta abierta. Abre el archivo HTML en tu navegador.")
