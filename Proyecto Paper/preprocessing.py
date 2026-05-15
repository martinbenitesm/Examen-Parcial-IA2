import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR     = "data/raw"
OUTPUT_DIR   = "data/processed"
DDOS_DIR     = os.path.join(BASE_DIR, "normal_DDoS")
NORMAL_DIR   = os.path.join(BASE_DIR, "normal_Bro")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Formato estándar de ToN_IoT para conn.csv
COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents"
]

NUMERIC_FEATURES = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
]

CATEGORICAL_FEATURES = ["proto", "service", "conn_state"]

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

# ─────────────────────────────────────────
# PASO 1: CARGAR CON DETECCIÓN DE CABECERAS
# ─────────────────────────────────────────
def load_csvs(folder, label):
    all_files = glob.glob(os.path.join(folder, "**", "conn.csv"), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No se encontraron conn.csv en: {folder}")
    
    dfs = []
    for f in sorted(all_files):
        try:
            # Detective de cabeceras: leemos 1 fila para ver si tiene nombres
            df_check = pd.read_csv(f, nrows=1, low_memory=False)
            
            # Si no detectamos 'duration', forzamos los nombres estándar
            if "duration" not in df_check.columns:
                df = pd.read_csv(f, header=None, names=COLUMNS, low_memory=False)
            else:
                df = pd.read_csv(f, low_memory=False)
            
            df["label"] = label
            dfs.append(df)
            log(f"  ✓ Cargado: {f}  ({len(df):,} filas)")
        except Exception as e:
            log(f"  ✗ Error al cargar {f}: {e}")
            
    return pd.concat(dfs, ignore_index=True)

log("=" * 60)
log("PASO 1: CARGANDO DATOS (SISTEMA HÍBRIDO)")
log("=" * 60)

df_ddos = load_csvs(DDOS_DIR, label=1)
df_normal = load_csvs(NORMAL_DIR, label=0)
df = pd.concat([df_ddos, df_normal], ignore_index=True)

# ─────────────────────────────────────────
# PASO 2: LIMPIEZA PROFESIONAL
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("PASO 2: LIMPIEZA Y TRATAMIENTO")
log("=" * 60)

# Solo nos quedamos con lo necesario
FEATURES_TO_KEEP = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["label"]
df = df[[c for c in FEATURES_TO_KEEP if c in df.columns]]

df.replace("-", np.nan, inplace=True)

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    median_val = df[col].median()
    # Llenar nulos con la mediana (más robusto que 0)
    df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].fillna("unknown")

# NOTA: No aplicamos drop_duplicates aquí para mantener el volumen total solicitado
log(f"Filas totales listas para balancear: {len(df):,}")

# ─────────────────────────────────────────
# PASO 3: BALANCEO 50/50 REAL
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("PASO 3: BALANCEO DE CLASES (SUBMUESTREO)")
log("=" * 60)

df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

n_min = min(len(df_0), len(df_1))

df_resampled = pd.concat([
    df_0.sample(n=n_min, random_state=42),
    df_1.sample(n=n_min, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

log(f"Distribución final: Normal ({n_min:,}) | DDoS ({n_min:,})")

# ─────────────────────────────────────────
# PASO 4: CODIFICACIÓN Y NORMALIZACIÓN
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("PASO 4: PROCESADO FINAL Y EXPORTACIÓN DE REGLAS")
log("=" * 60)

# 1. Guardar un Encoder independiente por cada columna
diccionario_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df_resampled[col] = le.fit_transform(df_resampled[col].astype(str))
    diccionario_encoders[col] = le 

# 2. Escalar y guardar el Scaler Maestro
scaler = MinMaxScaler()
df_resampled[NUMERIC_FEATURES + CATEGORICAL_FEATURES] = scaler.fit_transform(df_resampled[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

# ─────────────────────────────────────────
# PASO 5: GUARDADO
# ─────────────────────────────────────────
output_path = os.path.join(OUTPUT_DIR, "dataset_processed.csv")
df_resampled.to_csv(output_path, index=False)

# ¡NUEVO!: Exportamos las reglas matemáticas para producción
joblib.dump(diccionario_encoders, "diccionario_encoders.pkl")
joblib.dump(scaler, "scaler_maestro.pkl")

# Guardar reporte (tu código original)
report_path = os.path.join(OUTPUT_DIR, "reporte_preprocesamiento.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log(f"\nPREPROCESAMIENTO EXITOSO")
log("Artefactos guardados: diccionario_encoders.pkl y scaler_maestro.pkl")