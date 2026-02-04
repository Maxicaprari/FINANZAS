
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Desactivar warnings SSL
requests.packages.urllib3.disable_warnings()

# ============================
# CONFIGURACIÓN
# ============================

# Mapeo de columnas de tu base a IDs de la API del BCRA
# Estos son los IDs más comunes que suelen funcionar

MAPEO_VARIABLES = {
    # Reservas
    'reservas_internacionales_bcra_saldos': 1,
    
    # Tipo de cambio
    'tipo_de_cambio_moneda_de_cada_momento': 4,
    'tipo_de_cambio_en_pesos_equivalentes': 5,
    
    # Agregados monetarios
    'agregados_monetarios_en_pesos_bm': 8,
    'agregados_monetarios_en_pesos_bym_en_circulacion': 10,
    
    # Tasas (PASES, LELIQ, BADLAR)
    'tasa_interes_prestamos_interfinancieros_hasta_15d_pesos': 18,  # PASES aproximado
    
    # Préstamos
    'prestamos_al_sector_privado_pesos': 19,
    'prestamos_al_sector_privado_usd': 20,
}

# ============================
# FUNCIONES DE CARGA
# ============================

def cargar_base_existente(ruta_archivo):
    """
    Carga tu base histórica existente
    """
    print("📂 Cargando base existente...")
    
    # Intentar diferentes formatos
    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo.endswith('.xlsx'):
        df = pd.read_excel(ruta_archivo)
    elif ruta_archivo.endswith('.parquet'):
        df = pd.read_parquet(ruta_archivo)
    else:
        # Intentar CSV por defecto
        df = pd.read_csv(ruta_archivo)
    
    # Convertir fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    print(f"✅ Base cargada: {len(df)} observaciones")
    print(f"📅 Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"📊 Variables: {len(df.columns)}")
    
    return df


# ============================
# FUNCIONES DE DESCARGA API
# ============================

def descargar_variable_bcra(id_variable, fecha_desde, fecha_hasta=None):
    """
    Descarga una variable del BCRA usando v3.0/Monetarias
    """
    if fecha_hasta is None:
        fecha_hasta = datetime.now().strftime("%Y-%m-%d")
    
    # Convertir fechas al formato del BCRA
    if isinstance(fecha_desde, pd.Timestamp):
        fecha_desde = fecha_desde.strftime("%Y-%m-%d")
    
    url = f"https://api.bcra.gob.ar/estadisticas/v3.0/Monetarias/{id_variable}"
    
    try:
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        
        data_json = response.json()
        
        if "results" in data_json and len(data_json["results"]) > 0:
            df = pd.DataFrame(data_json["results"])
            
            if "fecha" in df.columns and "valor" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"])
                
                # Filtrar por rango de fechas
                df = df[(df['fecha'] >= fecha_desde) & (df['fecha'] <= fecha_hasta)]
                
                return df[['fecha', 'valor']]
        
    except Exception as e:
        print(f"   ⚠️ Error descargando ID {id_variable}: {str(e)[:50]}")
    
    return None


def descargar_variables_nuevas(mapeo_vars, fecha_desde, fecha_hasta=None):
    """
    Descarga múltiples variables nuevas desde fecha_desde
    """
    print(f"\n📥 Descargando datos nuevos desde {fecha_desde}...")
    print("=" * 70)
    
    datos_nuevos = {}
    
    for nombre_col, id_var in mapeo_vars.items():
        print(f"Descargando {nombre_col[:40]}... (ID: {id_var})", end=" ")
        
        df = descargar_variable_bcra(id_var, fecha_desde, fecha_hasta)
        
        if df is not None and not df.empty:
            df = df.rename(columns={'valor': nombre_col})
            datos_nuevos[nombre_col] = df
            print(f"✅ {len(df)} nuevas obs")
        else:
            print(f"❌")
    
    if not datos_nuevos:
        print("\n⚠️ No se descargaron datos nuevos")
        return pd.DataFrame()
    
    # Consolidar todas las variables nuevas
    print(f"\n🔗 Consolidando {len(datos_nuevos)} variables...")
    
    df_nuevo = None
    for nombre_col, df_var in datos_nuevos.items():
        if df_nuevo is None:
            df_nuevo = df_var
        else:
            df_nuevo = pd.merge(df_nuevo, df_var, on='fecha', how='outer')
    
    df_nuevo = df_nuevo.sort_values('fecha').reset_index(drop=True)
    
    print(f"✅ Datos nuevos consolidados: {len(df_nuevo)} observaciones")
    
    return df_nuevo


# ============================
# FUNCIONES DE CONCATENACIÓN
# ============================

def concatenar_bases(df_historico, df_nuevo):
    """
    Concatena la base histórica con los datos nuevos
    Elimina duplicados y ordena cronológicamente
    """
    print("\n🔗 Concatenando bases...")
    print("=" * 70)
    
    if df_nuevo.empty:
        print("⚠️ No hay datos nuevos para concatenar")
        return df_historico
    
    # Verificar columnas comunes
    columnas_comunes = set(df_historico.columns) & set(df_nuevo.columns)
    print(f"📊 Columnas comunes: {len(columnas_comunes)}")
    
    if len(columnas_comunes) <= 1:  # Solo 'fecha'
        print("⚠️ Pocas columnas en común. Agregando datos nuevos como columnas adicionales...")
        df_completo = pd.merge(df_historico, df_nuevo, on='fecha', how='outer')
    else:
        # Concatenar verticalmente
        df_completo = pd.concat([df_historico, df_nuevo], ignore_index=True)
    
    # Eliminar duplicados (quedarse con el más reciente)
    antes = len(df_completo)
    df_completo = df_completo.sort_values('fecha').drop_duplicates(subset='fecha', keep='last')
    despues = len(df_completo)
    
    if antes > despues:
        print(f"✅ Eliminados {antes - despues} duplicados")
    
    # Ordenar cronológicamente
    df_completo = df_completo.sort_values('fecha').reset_index(drop=True)
    
    print(f"✅ Base consolidada: {len(df_completo)} observaciones totales")
    print(f"📅 Nuevo período: {df_completo['fecha'].min().date()} → {df_completo['fecha'].max().date()}")
    
    return df_completo


# ============================
# FUNCIONES DE ANÁLISIS
# ============================

def analizar_actualizacion(df_original, df_actualizado):
    """
    Analiza qué se agregó en la actualización
    """
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DE ACTUALIZACIÓN")
    print("=" * 70)
    
    print(f"\nObservaciones:")
    print(f"  Original:     {len(df_original):5d}")
    print(f"  Actualizado:  {len(df_actualizado):5d}")
    print(f"  Nuevas:       {len(df_actualizado) - len(df_original):5d}")
    
    print(f"\nVariables:")
    print(f"  Original:     {len(df_original.columns):3d}")
    print(f"  Actualizado:  {len(df_actualizado.columns):3d}")
    
    # Variables nuevas
    cols_nuevas = set(df_actualizado.columns) - set(df_original.columns)
    if cols_nuevas:
        print(f"\n📌 Variables nuevas agregadas ({len(cols_nuevas)}):")
        for col in sorted(cols_nuevas):
            if col != 'fecha':
                print(f"  • {col}")
    
    # Rango de fechas
    fecha_max_original = df_original['fecha'].max()
    fecha_max_actualizado = df_actualizado['fecha'].max()
    
    if fecha_max_actualizado > fecha_max_original:
        dias_nuevos = (fecha_max_actualizado - fecha_max_original).days
        print(f"\n📅 Período extendido:")
        print(f"  Última fecha original:    {fecha_max_original.date()}")
        print(f"  Última fecha actualizada: {fecha_max_actualizado.date()}")
        print(f"  Días agregados:           {dias_nuevos}")


def verificar_calidad_datos(df):
    """
    Verifica la calidad de los datos
    """
    print("\n" + "=" * 70)
    print("🔍 VERIFICACIÓN DE CALIDAD DE DATOS")
    print("=" * 70)
    
    total_obs = len(df)
    
    print(f"\nCobertura por variable:")
    for col in df.columns:
        if col != 'fecha':
            no_nulos = df[col].notna().sum()
            pct = (no_nulos / total_obs) * 100
            
            if pct < 50:
                emoji = "⚠️"
            elif pct < 80:
                emoji = "📊"
            else:
                emoji = "✅"
            
            print(f"  {emoji} {col[:50]:50s}: {pct:5.1f}% ({no_nulos}/{total_obs})")


def guardar_base_completa(df, nombre='df_bcra_completo'):
    """
    Guarda la base completa en múltiples formatos
    """
    print(f"\n💾 Guardando base completa...")
    
    # CSV
    archivo_csv = f'{nombre}.csv'
    df.to_csv(archivo_csv, index=False)
    print(f"  ✅ {archivo_csv}")
    
    # Excel
    try:
        archivo_excel = f'{nombre}.xlsx'
        df.to_excel(archivo_excel, index=False)
        print(f"  ✅ {archivo_excel}")
    except Exception as e:
        print(f"  ⚠️ No se pudo guardar Excel: {str(e)[:50]}")
    
    # Parquet
    try:
        archivo_parquet = f'{nombre}.parquet'
        df.to_parquet(archivo_parquet, index=False)
        print(f"  ✅ {archivo_parquet}")
    except Exception as e:
        print(f"  ⚠️ No se pudo guardar Parquet: {str(e)[:50]}")


# ============================
# EJECUCIÓN PRINCIPAL
# ============================

def actualizar_base_bcra(ruta_base_existente, guardar=True):
    """
    Función principal que actualiza la base del BCRA
    
    Parámetros:
    - ruta_base_existente: path al archivo con tu base histórica
    - guardar: si True, guarda el resultado en archivos
    
    Retorna:
    - df_bcra_completo: DataFrame consolidado y actualizado
    """
    
    print("=" * 70)
    print("🇦🇷 ACTUALIZADOR DE BASE BCRA")
    print("=" * 70)
    
    # 1. Cargar base existente
    df_historico = cargar_base_existente(ruta_base_existente)
    
    # 2. Identificar última fecha
    ultima_fecha = df_historico['fecha'].max()
    fecha_desde = ultima_fecha + timedelta(days=1)  # Desde el día siguiente
    fecha_hasta = datetime.now()
    
    print(f"\n📅 Buscando datos desde: {fecha_desde.date()} hasta: {fecha_hasta.date()}")
    
    # 3. Descargar datos nuevos
    df_nuevo = descargar_variables_nuevas(MAPEO_VARIABLES, fecha_desde, fecha_hasta)
    
    # 4. Concatenar bases
    df_bcra_completo = concatenar_bases(df_historico, df_nuevo)
    
    # 5. Análisis
    analizar_actualizacion(df_historico, df_bcra_completo)
    verificar_calidad_datos(df_bcra_completo)
    
    # 6. Guardar
    if guardar:
        guardar_base_completa(df_bcra_completo)
    
    print("\n" + "=" * 70)
    print("✅ ACTUALIZACIÓN COMPLETADA")
    print("=" * 70)
    
    return df_bcra_completo


# ============================
# EJECUCIÓN
# ============================

if __name__ == "__main__":
    
    # IMPORTANTE: Cambia esta ruta por la ubicación de tu archivo
    RUTA_TU_BASE = "df_BCRA.xlsx"  # O .xlsx, .parquet
    
    # Opción 1: Ejecutar con prompt interactivo
    print("🔧 Configuración:")
    print(f"   Ruta actual: {RUTA_TU_BASE}")
    
    usar_ruta = input("\n¿Está correcta la ruta? (s/n): ").lower()
    
    if usar_ruta != 's':
        RUTA_TU_BASE = input("Ingresa la ruta correcta de tu base: ")
    
    # Ejecutar actualización
    df_bcra_completo = actualizar_base_bcra(RUTA_TU_BASE, guardar=True)
    
    # Mostrar resultado
    print("\n📊 Vista previa de df_bcra_completo:")
    print(df_bcra_completo.info())
    print("\n🔍 Primeras filas:")
    print(df_bcra_completo.head())
    print("\n🔍 Últimas filas:")
    print(df_bcra_completo.tail())
    
    print("\n" + "=" * 70)
    print("💡 El DataFrame 'df_bcra_completo' está listo para usar")
    print("=" * 70)
    
    # Ejemplo de uso posterior
    print("\n📝 Ejemplo de uso:")
    print("""
    # Para análisis posterior:
    df = df_bcra_completo.copy()
    
    # Imputar NaNs
    df = df.fillna(method='ffill')  # Forward fill
    
    # Normalizar para MST/PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df.select_dtypes(include=[np.number])),
        columns=df.select_dtypes(include=[np.number]).columns
    )
    
    # Concatenar con API 912
    df_912 = pd.read_csv('api912_datos.csv')
    df_final = pd.merge(df_bcra_completo, df_912, on='fecha', how='outer')
    """)
