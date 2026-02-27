import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_detectar_sobreajuste():
    
    n = random.randint(80, 150)
    d = random.randint(3,6)
    
    X = np.random.randn(n,d)
    y = np.random.randint(0,2,n)
    
    model1 = DecisionTreeClassifier(max_depth=None, random_state=42)
    model2 = DecisionTreeClassifier(max_depth=2, random_state=42)
    
    model1.fit(X,y)
    model2.fit(X,y)
    
    acc1 = accuracy_score(y, model1.predict(X))
    acc2 = accuracy_score(y, model2.predict(X))
    
    resultado = (acc1 - acc2) > 0.15
    
    return {"X":X.copy(),"y":y.copy()}, resultado


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_sobreajuste()
    
    print("===== INPUT =====")
    print("X completo:")
    print(entrada["X"])
    print("\nShape X:", entrada["X"].shape)
    print("\ny completo:")
    print(entrada["y"])
    print("Shape y:", entrada["y"].shape)
    
    print("\n===== OUTPUT ESPERADO =====")
    print("¿Existe sobreajuste?:", salida)
