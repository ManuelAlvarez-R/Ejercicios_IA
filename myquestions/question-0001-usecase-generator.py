import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_detectar_sesiones_irregulares():
    
    n_users = random.randint(5, 12)
    registros = []
    
    for user in range(n_users):
        n_sesiones = random.randint(4, 8)
        base_time = pd.Timestamp("2024-01-01")
        duraciones = np.random.randint(10, 60, n_sesiones)
        
        for i in range(n_sesiones):
            registros.append([
                user,
                base_time + pd.DateOffset(hours=i*3),
                duraciones[i]
            ])
    
    df = pd.DataFrame(registros, columns=["usuario_id","timestamp","duracion_minutos"])
    
    umbral = random.randint(20, 50)
    
    usuarios_irregulares = []
    
    for user, grupo in df.groupby("usuario_id"):
        grupo_ordenado = grupo.sort_values("timestamp")
        acumulado = grupo_ordenado["duracion_minutos"].cumsum()
        difs = acumulado.diff().abs()
        if (difs > umbral*2).any():
            usuarios_irregulares.append(user)
    
    usuarios_irregulares = sorted(usuarios_irregulares)
    
    return {"df": df.copy(), "umbral_minutos": umbral}, usuarios_irregulares


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_sesiones_irregulares()
    
    print("=== INPUT ===")
    print("Umbral:", entrada["umbral_minutos"])
    print(entrada["df"].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    print(salida)
