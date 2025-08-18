import requests
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


ids_interes = [1, 4, 5, 18, 19, 22, 23]

variables_info = {
    1: "Reservas Internacionales",
    4: "Tipo de Cambio Minorista ($/USD)",
    5: "Tipo de Cambio Mayorista ($/USD)",
    18: "Base Monetaria",
    19: "Circulante en poder del público",
    22: "Depósitos del sector privado en pesos",
    23: "Crédito al sector privado en pesos"
}

# df_info = pd.DataFrame(list(variables_info.items()), columns=["ID", "Descripción"])
# print(" Variables seleccionadas:")
# display(df_info)


##### COMPONENTES PRINCIPALES + GRÁFICOS 

series_dict = {}

for var_id in ids_interes:
    url = f"https://api.bcra.gob.ar/estadisticas/v3.0/Monetarias/{var_id}"
    r = requests.get(url, verify=False)

    try:
        data_json = r.json()
        if "results" in data_json:
            data = pd.DataFrame(data_json["results"])
            if not data.empty and "fecha" in data.columns and "valor" in data.columns:
                data["fecha"] = pd.to_datetime(data["fecha"])
                desc = f"Var_{var_id}"
                data = data.set_index("fecha")[["valor"]].rename(columns={"valor": desc})
                series_dict[desc] = data
                print(f" Descargada variable {var_id}")
            else:
                print(f" Variable {var_id} no tiene datos válidos.")
        else:
            print(f" Variable {var_id} no tiene 'results'.")
    except Exception as e:
        print(f" Error con variable {var_id}: {e}")


if not series_dict:
    raise ValueError("No se pudo descargar ninguna serie.")

df = pd.concat(series_dict.values(), axis=1)
df = df.dropna(axis=1, how="all").dropna()
print("Shape del DataFrame combinado:", df.shape)
print(df.head())


X_scaled = StandardScaler().fit_transform(df)


pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print("\nVarianza explicada por los primeros componentes:")
for i, var in enumerate(explained, 1):
    print(f"Componente {i}: {var:.2%}")


plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained)+1), explained.cumsum(), marker="o")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("Scree Plot - PCA")
plt.grid()
plt.show()


pc1 = X_pca[:,0]
pc2 = X_pca[:,1]

plt.figure(figsize=(8,6))
plt.scatter(pc1, pc2, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot PCA (Observaciones + Variables)")

loadings = pca.components_.T[:, :2]  # cargas de PC1 y PC2
for i, var in enumerate(df.columns):
    plt.arrow(0, 0, loadings[i,0]*5, loadings[i,1]*5, 
              color="r", alpha=0.5, head_width=0.05)
    plt.text(loadings[i,0]*5.2, loadings[i,1]*5.2, var, fontsize=8)

plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.grid()
plt.show()




import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación entre variables")
plt.show()


df.plot(subplots=True, figsize=(12,10), title="Series Temporales")
plt.tight_layout()
plt.show()


df_rolling = df.rolling(window=30).mean()
df_rolling.plot(figsize=(12,6))
plt.title("Promedio móvil (30 días)")
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=range(len(df)), cmap="viridis", alpha=0.7)
plt.colorbar(label="Índice temporal")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proyección temporal en componentes principales")
plt.show()


loading_matrix = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(df.columns))],
    index=df.columns
)

print("Importancia de las variables en los primeros componentes:")
display(loading_matrix.iloc[:,:2].sort_values("PC1", ascending=False))


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
labels = kmeans.labels_

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="Set1", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusterización de series (KMeans en espacio PCA)")
plt.show()


# df.to_csv("bcra_series_procesadas.csv")
# print(" Archivo guardado: bcra_series_procesadas.csv")





