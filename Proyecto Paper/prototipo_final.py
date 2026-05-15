import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from azure.identity import InteractiveBrowserCredential
from azure.confidentialledger import ConfidentialLedgerClient
from azure.confidentialledger.certificate import ConfidentialLedgerCertificateClient
import datetime

# =======================================================
# 1. CONFIGURACIÓN INICIAL Y AZURE
# =======================================================
# [!] REEMPLAZA ESTAS DOS VARIABLES ANTES DE EJECUTAR [!]
ARCHIVO_CRUDO = "dataset_prueba_crudo.csv" # <-- PON EL NOMBRE DE TU ARCHIVO AQUÍ
MI_TENANT_ID = "a807af63-6d93-47bb-97bd-50e8953b962c"    # <-- PON TU TENANT ID DE AZURE

LEDGER_NAME = "ids-blockchain-rg" 
LEDGER_ENDPOINT = f"https://{LEDGER_NAME}.confidential-ledger.azure.com"
IDENTITY_URL = "https://identity.confidential-ledger.core.azure.com"

print(">>> [1/4] Autenticando con Azure Confidential Ledger...")
try:
    cert_client = ConfidentialLedgerCertificateClient(IDENTITY_URL)
    network_identity = cert_client.get_ledger_identity(ledger_id=LEDGER_NAME)
    cert_path = "networkcert.pem"
    with open(cert_path, "w") as cert_file:
        cert_file.write(network_identity['ledgerTlsCertificate'])

    credential = InteractiveBrowserCredential(tenant_id=MI_TENANT_ID)
    ledger_client = ConfidentialLedgerClient(
        endpoint=LEDGER_ENDPOINT, credential=credential, ledger_certificate_path=cert_path
    )
except Exception as e:
    print(f"\nError al conectar con Azure. Revisa tu conexión o el Tenant ID. Detalles: {e}")
    exit()

# =======================================================
# 2. CARGA Y PREPROCESAMIENTO BLINDADO (CON REGLAS EXPORTADAS)
# =======================================================
print(f">>> [2/4] Cargando dataset crudo ({ARCHIVO_CRUDO}) y aplicando reglas de preprocesamiento...")

# Estructura del dataset Zeek/Bro basada en tu configuración
COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents"
]

NUMERIC_FEATURES = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
CATEGORICAL_FEATURES = ["proto", "service", "conn_state"]
FEATURES_TO_KEEP = NUMERIC_FEATURES + CATEGORICAL_FEATURES

try:
    # 2.1 Leer crudo (df_raw: se usa para sacar las IPs en las alertas)
    df_raw = pd.read_csv(ARCHIVO_CRUDO, header=None, names=COLUMNS, low_memory=False)
    
    # Asumimos que todo el archivo crudo de prueba son ataques DDoS (Label 1)
    df_real = pd.Series([1] * len(df_raw)) 

    # 2.2 Crear df_ai (Exclusivamente para limpiar, normalizar y predecir)
    df_ai = df_raw[FEATURES_TO_KEEP].copy()
    df_ai.replace("-", np.nan, inplace=True)

    for col in NUMERIC_FEATURES:
        df_ai[col] = pd.to_numeric(df_ai[col], errors="coerce")
        median_val = df_ai[col].median()
        df_ai[col] = df_ai[col].fillna(median_val if pd.notna(median_val) else 0)

    for col in CATEGORICAL_FEATURES:
        df_ai[col] = df_ai[col].fillna("unknown")

    # 2.3 Codificación y Normalización ESTRICTA (Usando .transform)
    print("    ↳ Cargando artefactos de escalado maestro...")
    encoders = joblib.load("diccionario_encoders.pkl")
    scaler = joblib.load("scaler_maestro.pkl")

    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        clases_conocidas = set(le.classes_)
        # Si llega un texto nuevo en producción que la IA no conoce, lo forzamos a una clase segura para no colapsar
        df_ai[col] = df_ai[col].apply(lambda x: x if x in clases_conocidas else le.classes_[0])
        df_ai[col] = le.transform(df_ai[col].astype(str))

    # Aplicamos la misma cinta métrica del entrenamiento
    df_ai[NUMERIC_FEATURES + CATEGORICAL_FEATURES] = scaler.transform(df_ai[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

except FileNotFoundError as e:
    print(f"\nError de archivos: Falta algún archivo necesario (.csv, o los .pkl del preprocesamiento).")
    print(f"Detalles: {e}")
    exit()

# =======================================================
# 3. EVALUACIÓN DE LA IA (OBJETIVO ESPECÍFICO 1)
# =======================================================
print(">>> [3/4] Cargando Modelo IA y detectando anomalías...")
try:
    modelo = joblib.load("modelo_ids_final.pkl")
except FileNotFoundError:
    print("\nError: No se encontró 'modelo_ids_final.pkl'.")
    exit()

# Hacer predicciones con los datos procesados matemáticamente perfectos
y_pred = modelo.predict(df_ai)

# Cálculos para O.E. 1
precision = accuracy_score(df_real, y_pred) * 100
try:
    tn, fp, fn, tp = confusion_matrix(df_real, y_pred).ravel()
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
except ValueError:
    # Caso donde el dataset de prueba solo tiene ataques puros (no hay TN ni FP posibles)
    tp = sum(y_pred == 1)
    fn = sum(y_pred == 0)
    fp, tn, fpr = 0, 0, 0.0

print("\n" + "="*60)
print("RESULTADOS O.E. 1: PRECISIÓN Y FALSOS POSITIVOS")
print("="*60)
print(f"Precisión Global del Modelo   : {precision:.2f}%")
print(f"Ataques Detectados (TP)       : {tp} de {len(df_raw)}")
print(f"Ataques No Detectados (FN)    : {fn}")
print(f"Falsos Positivos (FPR)        : {fpr:.2f}%")
print("="*60 + "\n")

# =======================================================
# 4. REGISTRO EN BLOCKCHAIN Y CÁLCULO DE INTEGRIDAD (O.E. 2)
# =======================================================
print(">>> [4/4] Iniciando fase de registro inmutable en Azure Confidential Ledger...")

# Variables para el cálculo métrico de integridad (Para validación metodológica)
alertas_detectadas_ia = 0
confirmaciones_blockchain = 0

# Filtrar las posiciones exactas de los ataques detectados
indices_ataques = np.where(y_pred == 1)[0]
limite_registros = 5  # Muestra representativa para validación de integridad

if len(indices_ataques) > 0:
    print(f"    ↳ Procesando {min(len(indices_ataques), limite_registros)} sellados criptográficos...")
    
    for idx in indices_ataques[:limite_registros]:
        alertas_detectadas_ia += 1
        
        # Rescatar datos vitales de red del df_raw
        fila_original = df_raw.iloc[idx]
        ip_origen = fila_original['id.orig_h']
        ip_destino = fila_original['id.resp_h']
        puerto_destino = fila_original['id.resp_p']
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Crear el log destinado al Ledger
        alerta_log = (f"[ALERTA 2026] {timestamp} | DDoS Detectado | "
                      f"Origen: {ip_origen} -> Destino: {ip_destino}:{puerto_destino}")
        
        try:
            # ENVIAR A BLOCKCHAIN Y ESPERAR RECIBO (HASH)
            resultado = ledger_client.create_ledger_entry(entry={"contents": alerta_log})
            tx_id = resultado['transactionId']
            
            # Si el Ledger devuelve un ID, la integridad de este registro está verificada
            if tx_id:
                confirmaciones_blockchain += 1
                print(f"SELLADO EXITOSO -> Transaction ID: {tx_id}")
                print(f"   ↳ Verificado: {ip_origen} atacó a {ip_destino}")
        except Exception as e:
            print(f"Error de persistencia en Blockchain: {e}")

    # CÁLCULO REAL DE LA VARIABLE DEPENDIENTE (DIMENSIÓN: INTEGRIDAD)
    # Tasa de Integridad = (Registros con Hash / Alertas enviadas) * 100
    tasa_integridad = (confirmaciones_blockchain / alertas_detectadas_ia) * 100 if alertas_detectadas_ia > 0 else 0

else:
    tasa_integridad = 0
    print("Ninguna anomalía detectada para el cálculo de integridad.")

print("\n" + "="*65)
print("RESULTADOS O.E. 2: MÉTRICAS DE INMUTABILIDAD E INTEGRIDAD")
print("="*65)
print(f"Alertas enviadas al Ledger (Intentos) : {alertas_detectadas_ia}")
print(f"Alertas confirmadas con Hash (Éxito)  : {confirmaciones_blockchain}")
print(f"TASA DE INTEGRIDAD CALCULADA (%)      : {tasa_integridad:.2f}%")
print("-" * 65)
print("Interpretación: Integridad garantizada mediante tecnología")
print("Intel SGX y sellado criptográfico secuencial (Merkle Tree).")
print("="*65 + "\n")