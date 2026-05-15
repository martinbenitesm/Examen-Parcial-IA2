import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import os

# Crear carpeta para resultados si no existe
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Cargar el dataset procesado
print(">>> Cargando dataset balanceado...")
df = pd.read_csv("data/processed/dataset_processed.csv")

# 2. Separar características (X) y etiqueta (y)
X = df.drop('label', axis=1)
y = df['label']

# 3. División Entrenamiento/Prueba (80/20)
# Usamos stratify para mantener el balance 50/50 en ambos sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Entrenar Random Forest
print(">>> Entrenando modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluación y Predicciones
y_pred = model.predict(X_test)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# 6. Mostrar Resultados con Decimales Reales (Para el texto del Paper)
print("\n" + "="*45)
print("MÉTRICAS TÉCNICAS (SIN REDONDEO)")
print("="*45)
for label in ['0', '1']:
    tipo = "Normal (0)" if label == '0' else "DDoS (1)"
    m = report_dict[label]
    print(f"CLASE {tipo}:")
    print(f"  - Precision: {m['precision']:.6f}")
    print(f"  - Recall:    {m['recall']:.6f}")
    print(f"  - F1-Score:  {m['f1-score']:.6f}")

print("-" * 45)
print(f"ACCURACY GLOBAL: {accuracy_score(y_test, y_pred):.6f}")
print("="*45)

# 7. Generar Matriz de Confusión Visual (Para la imagen del Paper)
print("\n>>> Generando matriz de confusión visual...")
cm = confusion_matrix(y_test, y_pred)

# Convertir a porcentajes (Normalizado por fila para ver el Recall real)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(9, 7))
sns.set_theme(style="white")

# Dibujar Heatmap
sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Normal', 'DDoS'], 
            yticklabels=['Normal', 'DDoS'],
            annot_kws={"size": 14, "weight": "bold"})

plt.ylabel('Clase Real', fontsize=12, fontweight='bold')
plt.xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión: Clasificación de Tráfico ToN_IoT', fontsize=14, pad=20)

# Guardar imagen en alta resolución (300 DPI)
plt.tight_layout()
plt.savefig("results/matriz_confusion_final.png", dpi=300)
print("Imagen guardada en: results/matriz_confusion_final.png")

# 8. Guardar el modelo
joblib.dump(model, "models/modelo_ids_final.pkl")
print("Modelo binario guardado en: models/modelo_ids_final.pkl")

plt.show()