import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_detectar_secuencia_alternante():
    
    n_sensores = random.randint(4, 8)
    registros = []
    sensores_validos = []
    
    for s in range(n_sensores):
        n_obs = random.randint(4, 7)
        alternante = random.choice([True, False])
        
        if alternante:
            inicio = random.choice([0,1])
            estados = [(inicio+i)%2 for i in range(n_obs)]
            sensores_validos.append(s)
        else:
            estados = np.random.randint(0,2,n_obs)
        
        for i in range(n_obs):
            registros.append([s, i, estados[i]])
    
    df = pd.DataFrame(registros, columns=["sensor_id","orden","estado"])
    sensores_validos = sorted(sensores_validos)
    
    return {"df": df.copy()}, sensores_validos


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_secuencia_alternante()
    
    print("===== INPUT =====")
    print(entrada["df"])
    
    print("\n===== OUTPUT ESPERADO =====")
    print(salida)
