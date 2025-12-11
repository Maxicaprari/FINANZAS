import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# PARÁMETROS OPTIMIZADOS DE MERV 

MERV_PARAMS = {
    'kappa': 14.8770,
    'theta': 0.000091,
    'xi': 0.0504,
    'rho': -0.4926,
    'v0': 0.005179
}

print("="*60)
print("VISUALIZACIONES RÁPIDAS - MERV")
print("Usando parámetros optimizados")
print("="*60)
print(f"κ = {MERV_PARAMS['kappa']:.4f}")
print(f"θ = {MERV_PARAMS['theta']:.6f}")
print(f"ξ = {MERV_PARAMS['xi']:.4f}")
print(f"ρ = {MERV_PARAMS['rho']:.4f}")
print(f"V₀ = {MERV_PARAMS['v0']:.6f}")
print("="*60)


print("\n[1/4] Descargando datos históricos de ^MERV...")
datos = yf.download("^MERV", period='5y', progress=False)
precios = datos['Close'].dropna()
retornos = np.log(precios / precios.shift(1)).dropna()
print(f"✓ Datos descargados: {len(precios)} observaciones")

print("\n[2/4] Estimando volatilidad GARCH(1,1)...")
modelo_garch = arch_model(retornos * 100, vol='Garch', p=1, q=1, dist='normal')
fit_garch = modelo_garch.fit(disp='off', show_warning=False)
var_garch = (fit_garch.conditional_volatility / 100)**2
print("✓ GARCH estimado")


print("\n[3/4] Generando trayectoria Heston (determinística)...")
var_heston = np.zeros(len(var_garch))
v = MERV_PARAMS['v0']
dt = 1/252

for i in range(len(var_garch)):
    var_heston[i] = v
    dv = MERV_PARAMS['kappa'] * (MERV_PARAMS['theta'] - v) * dt
    v = max(v + dv, 1e-8)



fig, ax = plt.subplots(figsize=(14, 7))

vol_garch_anual = np.sqrt(var_garch) * np.sqrt(252) * 100
vol_heston_anual = np.sqrt(var_heston) * np.sqrt(252) * 100

ax.plot(precios.index[1:], vol_garch_anual,
        label='GARCH(1,1)', color='#2E86AB', alpha=0.7, linewidth=1.5)
ax.plot(precios.index[1:], vol_heston_anual,
        label='Heston (calibrado)', color='#A23B72', linewidth=2.5)

# Línea de theta (nivel de largo plazo)
theta_anual = np.sqrt(MERV_PARAMS['theta']) * np.sqrt(252) * 100
ax.axhline(theta_anual, color='#F18F01', linestyle='--', linewidth=2,
           label=f'θ (largo plazo) = {theta_anual:.1f}%', alpha=0.8)

ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatilidad Anualizada (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Volatilidades Estimadas - Índice MERVAL',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacion_volatilidades_merval.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado: comparacion_volatilidades_merval.png")
plt.show()  # Mostrar en pantalla
plt.close()


def heston_qe_rapido(S0, v0, kappa, theta, xi, rho, T=1.0, n_steps=252, n_paths=100):
    """Simulación QE optimizada"""
    np.random.seed(42)
    dt = T / n_steps
    psi_c = 1.5

    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    V[:, 0] = v0

    for i in range(n_steps):
        v = V[:, i]

        m = theta + (v - theta) * np.exp(-kappa * dt)
        s2 = (v * xi**2 * np.exp(-kappa * dt) / kappa) * (1 - np.exp(-kappa * dt)) + \
             (theta * xi**2 / (2 * kappa)) * (1 - np.exp(-kappa * dt))**2

        psi = np.where(m > 0, s2 / m**2, 0)

        mask_low = psi <= psi_c

        b2 = np.where(mask_low, 2/psi - 1 + np.sqrt(2/psi * (2/psi - 1)), 0)
        a = np.where(mask_low, m / (1 + b2), 0)
        z_v = np.random.normal(size=n_paths)
        V_low = a * (np.sqrt(b2) + z_v)**2

        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / m
        u = np.random.rand(n_paths)
        V_high = np.where(u <= p, 0, np.log((1 - p) / (1 - u)) / beta)

        V[:, i+1] = np.where(mask_low, V_low, V_high)
        V[:, i+1] = np.maximum(V[:, i+1], 1e-8)

        z1 = np.random.randn(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
        S[:, i+1] = S[:, i] * np.exp((0 - 0.5 * v) * dt + np.sqrt(v * dt) * z1)

    return S, V

print("\nSimulando trayectorias Heston QE...")
precio_inicial = float(precios.iloc[-1])
S_sim, V_sim = heston_qe_rapido(
    S0=precio_inicial,
    v0=MERV_PARAMS['v0'],
    kappa=MERV_PARAMS['kappa'],
    theta=MERV_PARAMS['theta'],
    xi=MERV_PARAMS['xi'],
    rho=MERV_PARAMS['rho'],
    T=1.0,
    n_steps=252,
    n_paths=100
)
print("✓ Simulaciones completadas")


fig, ax = plt.subplots(figsize=(14, 7))
time = np.linspace(0, 1, S_sim.shape[1])

for i in range(100):
    ax.plot(time, S_sim[i], alpha=0.15, color='#86BBD8', linewidth=0.8)

ax.plot(time, S_sim.mean(axis=0), 'k-', linewidth=3,
        label='Trayectoria Promedio', zorder=5)

precio_inicial = float(precios.iloc[-1])
ax.axhline(precio_inicial, color='#F18F01', linestyle='--',
           linewidth=2, label=f'Precio Inicial = {precio_inicial:,.0f}', alpha=0.8)

ax.set_xlabel('Tiempo (años)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precio (ARS)', fontsize=12, fontweight='bold')
ax.set_title('Trayectorias Simuladas del Precio - Modelo Heston QE para MERVAL\n(100 simulaciones, 1 año)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trayectorias_precio_merval.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado: trayectorias_precio_merval.png")
plt.show()  # Mostrar en pantalla
plt.close()

fig, ax = plt.subplots(figsize=(14, 7))

vol_sim = np.sqrt(V_sim) * np.sqrt(252) * 100


for i in range(100):
    ax.plot(time, vol_sim[i], alpha=0.15, color='#F4A259', linewidth=0.8)

ax.plot(time, vol_sim.mean(axis=0), 'k-', linewidth=3,
        label='Volatilidad Promedio', zorder=5)

v0_anual = np.sqrt(MERV_PARAMS['v0']) * np.sqrt(252) * 100
ax.axhline(v0_anual, color='#2E86AB', linestyle='--',
           linewidth=2, label=f'V (inicial) = {v0_anual:.1f}%', alpha=0.8)
ax.axhline(theta_anual, color='#A23B72', linestyle='--',
           linewidth=2, label=f'θ (largo plazo) = {theta_anual:.1f}%', alpha=0.8)

ax.set_xlabel('Tiempo', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatilidad Anualizada (%)', fontsize=12, fontweight='bold')
ax.set_title('Trayectorias Simuladas de Volatilidad - Modelo Heston QE para MERVAL\n(100 simulaciones, 1 año)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trayectorias_volatilidad_merval.png', dpi=300, bbox_inches='tight')

plt.show()  # Mostrar en pantalla
plt.close()

