import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def reducir_y_clasificar(X, y, varianza_deseada=0.95):

    # 1. Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Aplicar PCA conservando la varianza deseada
    pca = PCA(n_components=varianza_deseada, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    # Número de componentes seleccionados
    n_componentes = X_reduced.shape[1]

    # 3. Entrenar KNN
    modelo = KNeighborsClassifier()
    modelo.fit(X_reduced, y)

    # 4. Retornar resultados
    return n_componentes, modelo
