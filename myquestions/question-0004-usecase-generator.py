import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_comparar_svc_lineal_vs_rbf():
    
    n = random.randint(80,150)
    
    X = np.random.randn(n,2)
    y = (X[:,0]**2 + X[:,1] > 0).astype(int)
    
    model_linear = SVC(kernel="linear")
    model_rbf = SVC(kernel="rbf")
    
    model_linear.fit(X,y)
    model_rbf.fit(X,y)
    
    acc_lin = accuracy_score(y, model_linear.predict(X))
    acc_rbf = accuracy_score(y, model_rbf.predict(X))
    
    if acc_lin > acc_rbf:
        resultado = "linear"
    elif acc_rbf > acc_lin:
        resultado = "rbf"
    else:
        resultado = "empate"
    
    return {"X":X.copy(),"y":y.copy()}, resultado


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_comparar_svc_lineal_vs_rbf()
    
    print("===== INPUT =====")
    print("X:")
    print(entrada["X"])
    print("\ny:")
    print(entrada["y"])
    
    print("\n===== OUTPUT ESPERADO =====")
    print(salida)
