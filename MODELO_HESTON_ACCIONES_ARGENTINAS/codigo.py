import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

tickers_argentina = [
    "YPFD.BA", "PAMP.BA", "TGNO4.BA", "TGSU2.BA", "CEPU.BA", "METR.BA", "EDN.BA",
    "GGAL.BA", "BMA.BA", "BBAR.BA", "SUPV.BA",
    "TECO2.BA",
    "ALUA.BA", "CRES.BA", "LOMA.BA", "MIRG.BA", "AGRO.BA",
    "COME.BA", "VALO.BA", "SEMI.BA",
    "^MERV"
]


def descargar_datos(ticker, periodo='5y'):
    try:
        datos = yf.download(ticker, period=periodo, progress=False)
        if datos.empty or 'Close' not in datos.columns:
            return None
        precios = datos['Close'].dropna()
        if len(precios) < 100:
            return None
        return precios
    except:
        return None

def calcular_retornos_log(precios):
    return np.log(precios / precios.shift(1)).dropna()


def estimar_varianza_garch(retornos):
    try:
        modelo = arch_model(retornos * 100, vol='Garch', p=1, q=1, dist='normal')
        modelo_fit = modelo.fit(disp='off', show_warning=False)
        return (modelo_fit.conditional_volatility / 100)**2
    except:
        return None


def heston_likelihood(params, retornos, varianza_observada, dt=1/252):
    kappa, theta, xi, rho, v0 = params

    # Penalización por violar condición de Feller
    feller = 2*kappa*theta - xi**2
    if feller < 0:
        return 1e10

    if kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1 or v0 <= 0:
        return 1e10

    n = len(retornos)
    log_likelihood = 0.0
    v = v0

    for i in range(1, n):
        dv = kappa*(theta - v)*dt + xi*np.sqrt(v)*np.sqrt(dt)*np.random.randn()
        v_new = max(v + dv, 1e-8)

        v_obs = varianza_observada[i]
        log_likelihood += -0.5*np.log(2*np.pi*v_new) - 0.5*(v_obs - v_new)**2 / v_new

        v = v_new

    return -log_likelihood

def calibrar_heston(retornos, var_obs):
    dt = 1/252
    v_prom = np.mean(var_obs)
    v_var = np.var(var_obs)

    # Mejor estimación inicial
    x0 = np.array([
        3.0,                            # kappa
        v_prom,                         # theta
        min(np.sqrt(v_var)*np.sqrt(252), 0.8),    # xi
        -0.3,                           # rho
        var_obs[0]                      # v0
    ])

    bounds = [
        (0.5, 15),      # kappa
        (1e-6, 0.5),    # theta
        (0.05, 2.0),    # xi
        (-0.95, -0.05), # rho (negativo para leverage effect)
        (1e-8, 0.5)     # v0
    ]

    try:
        # Usar differential_evolution para mejor búsqueda global
        resultado = differential_evolution(
            lambda p: heston_likelihood(p, retornos, var_obs, dt),
            bounds,
            seed=42,
            maxiter=500,
            polish=True,
            workers=1
        )

        # Verificar condición de Feller
        kappa, theta, xi, rho, v0 = resultado.x
        feller = 2*kappa*theta - xi**2

        return resultado.x, feller
    except:
        return [None]*5, None


def heston_simulacion_qe(S0, v0, kappa, theta, xi, rho, r, T, n_steps, n_paths):
    dt = T/n_steps
    S = np.zeros((n_paths, n_steps+1))
    V = np.zeros((n_paths, n_steps+1))

    S[:, 0] = S0
    V[:, 0] = v0
    psi_c = 1.5

    for i in range(n_steps):
        for j in range(n_paths):
            v = V[j, i]

            m = theta + (v - theta)*np.exp(-kappa*dt)
            s2 = (v*xi**2*np.exp(-kappa*dt)/kappa)*(1 - np.exp(-kappa*dt)) + \
                 (theta*xi**2/(2*kappa))*(1 - np.exp(-kappa*dt))**2

            psi = s2/m**2 if m > 0 else 0

            if psi <= psi_c:
                b2 = 2/psi - 1 + np.sqrt(2/psi * (2/psi - 1))
                a = m/(1+b2)
                z = np.random.normal()
                V[j, i+1] = a*(np.sqrt(b2) + z)**2
            else:
                p = (psi - 1)/(psi + 1)
                beta = (1 - p)/m
                u = np.random.rand()
                if u <= p:
                    V[j, i+1] = 0
                else:
                    V[j, i+1] = np.log((1 - p)/(1 - u))/beta

            V[j, i+1] = max(V[j, i+1], 1e-8)

            z1 = np.random.randn()
            z2 = rho*z1 + np.sqrt(1 - rho**2)*np.random.randn()

            S[j, i+1] = S[j, i]*np.exp((r - 0.5*v)*dt + np.sqrt(v*dt)*z1)

    return S, V


def graficar_precio_historico(precios, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(precios.index, precios.values)
    plt.title(f"Precio Histórico - {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"precio_{ticker.replace('.BA','').replace('^','')}.png", dpi=150)
    plt.close()

def graficar_volatilidad(vol_garch, vol_heston, fechas, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(fechas, np.sqrt(vol_garch)*np.sqrt(252)*100, label='GARCH', alpha=0.7)
    plt.plot(fechas, np.sqrt(vol_heston)*np.sqrt(252)*100, label='Heston', alpha=0.7, linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Comparación de Volatilidades - {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad Anualizada (%)")
    plt.tight_layout()
    plt.savefig(f"comparacion_volatilidades_{ticker.replace('.BA','').replace('^','')}.png", dpi=150)
    plt.close()

def graficar_trayectorias_precio(S, T, ticker, S0):
    plt.figure(figsize=(12,6))
    time = np.linspace(0, T, S.shape[1])

    # Trayectorias individuales
    for i in range(min(100, S.shape[0])):
        plt.plot(time, S[i], alpha=0.3, color='lightblue', linewidth=0.5)

    # Trayectoria promedio
    plt.plot(time, S.mean(axis=0), 'k-', linewidth=2, label='Promedio')

    plt.title(f"Trayectorias Simuladas de Precio - {ticker}")
    plt.xlabel("Tiempo (años)")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"trayectorias_precio_{ticker.replace('.BA','').replace('^','')}.png", dpi=150)
    plt.close()

def graficar_trayectorias_volatilidad(V, T, ticker):
    plt.figure(figsize=(12,6))
    time = np.linspace(0, T, V.shape[1])

    # Trayectorias individuales
    for i in range(min(100, V.shape[0])):
        plt.plot(time, np.sqrt(V[i])*np.sqrt(252)*100, alpha=0.3, color='lightcoral', linewidth=0.5)

    # Trayectoria promedio
    plt.plot(time, np.sqrt(V.mean(axis=0))*np.sqrt(252)*100, 'k-', linewidth=2, label='Promedio')

    plt.title(f"Trayectorias Simuladas de Volatilidad - {ticker}")
    plt.xlabel("Tiempo (años)")
    plt.ylabel("Volatilidad Anualizada (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"trayectorias_volatilidad_{ticker.replace('.BA','').replace('^','')}.png", dpi=150)
    plt.close()

resultados = []

for ticker in tickers_argentina:
    print(f"Procesando: {ticker}")


    precios = descargar_datos(ticker)
    if precios is None:
        print("Error: no se pudieron descargar datos.")
        continue

    retornos = calcular_retornos_log(precios)
    var_garch = estimar_varianza_garch(retornos)

    if var_garch is None:
        print("Error: GARCH falló.")
        continue

    params, feller = calibrar_heston(retornos.values, var_garch)
    if None in params:
        print("Error: Heston falló.")
        continue

    kappa, theta, xi, rho, v0 = params

    print(f"Parámetros calibrados:")
    print(f"  kappa = {kappa:.4f}")
    print(f"  theta = {theta:.6f}")
    print(f"  xi = {xi:.4f}")
    print(f"  rho = {rho:.4f}")
    print(f"  v0 = {v0:.6f}")
    print(f"  Condición de Feller (2κθ - ξ²): {feller:.6f} {'✓' if feller > 0 else '✗ VIOLADA'}")

    resultados.append({
        "ticker": ticker,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "v0": v0,
        "feller": feller
    })

    graficar_precio_historico(precios, ticker)

    # Generación del proceso determinístico del Heston
    var_heston = np.zeros_like(var_garch)
    v = v0
    dt = 1/252
    for i in range(len(var_garch)):
        var_heston[i] = v
        dv = kappa*(theta - v)*dt
        v = max(v + dv, 1e-8)

    graficar_volatilidad(var_garch, var_heston, precios.index[1:], ticker)

    print("Simulando trayectorias para Heston QE...")
    S_sim, V_sim = heston_simulacion_qe(
        precios.iloc[-1], v0, kappa, theta, xi, rho,
        0.0, 1.0, 252, 100
    )

    graficar_trayectorias_precio(S_sim, 1.0, ticker, precios.iloc[-1])
    graficar_trayectorias_volatilidad(V_sim, 1.0, ticker)






df_final = pd.DataFrame(resultados)
