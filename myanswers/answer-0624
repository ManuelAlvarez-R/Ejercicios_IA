import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def clasificar_riesgo_psicologico(df, target_col):

    # Separar características y variable objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # Crear y entrenar modelo
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Calcular F1 Score
    resultado = f1_score(y_test, y_pred, zero_division=0)

    # Retornar float
    return float(resultado)
