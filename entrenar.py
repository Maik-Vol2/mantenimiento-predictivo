
# ============================================
# SISTEMA DE MANTENIMIENTO PREDICTIVO
# Script de entrenamiento del modelo ML
# ============================================

# 1. Importar las librerías necesarias
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 2. Cargar el dataset
print("Cargando dataset de sensores...")

df = pd.read_csv("sensores_planta.csv")


# 3. Limpiar los datos
print("Limpiando datos...")

df = df.dropna()

print(f"Registros disponibles: {len(df)}")


# 4. Separar variables de entrada y variable objetivo
X = df[[
    "temperatura_c",
    "vibracion_mms",
    "horas_operacion",
    "presion_psi"
]]

y = df["estado"]


# 5. Dividir los datos: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. Crear el modelo de Random Forest
print("Creando modelo de Machine Learning...")

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 7. Entrenar el modelo
print("Entrenando modelo...")

modelo.fit(X_train, y_train)


# 8. Realizar predicciones con los datos de prueba
y_pred = modelo.predict(X_test)


# 9. Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)

print("\n================================")
print(" RESULTADOS DEL ENTRENAMIENTO")
print("================================")

print(f"Precisión (Accuracy): {precision * 100:.2f}%")

if precision >= 0.80:
    print("Resultado: El modelo cumple con el mínimo del 80%.")
else:
    print("Resultado: El modelo no alcanza el mínimo del 80%.")


# 10. Guardar el modelo entrenado
joblib.dump(modelo, "modelo_mantenimiento.pkl")

print("\nModelo guardado correctamente como:")
print("modelo_mantenimiento.pkl")
