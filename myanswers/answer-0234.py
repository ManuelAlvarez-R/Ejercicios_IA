import pandas as pd

def alinear_sensor_eventos(
    sensor_df,
    eventos_df,
    time_col,
    value_col,
    event_time_col,
    direction="backward"
):
    
    # Copias para no modificar originales
    s = sensor_df.copy()
    e = eventos_df.copy()

    # 1. Convertir a datetime
    s[time_col] = pd.to_datetime(s[time_col], errors="coerce")
    e[event_time_col] = pd.to_datetime(e[event_time_col], errors="coerce")

    # 2. Eliminar fechas inválidas
    s = s.dropna(subset=[time_col])
    e = e.dropna(subset=[event_time_col])

    # 3. Ordenar
    s = s.sort_values(time_col).reset_index(drop=True)
    e = e.sort_values(event_time_col).reset_index(drop=True)

    # 4. merge_asof
    merged = pd.merge_asof(
        e.rename(columns={event_time_col: "event_time"}),
        s.rename(columns={
            time_col: "sensor_time",
            value_col: "sensor_value"
        }),
        left_on="event_time",
        right_on="sensor_time",
        direction=direction
    )

    # 5. lag en segundos
    merged["lag_seconds"] = (
        merged["event_time"] - merged["sensor_time"]
    ).dt.total_seconds()

    # Seleccionar columnas finales
    out = merged[
        ["event_time", "sensor_time", "sensor_value", "lag_seconds"]
    ]

    # Ordenar y resetear índice
    out = out.sort_values("event_time").reset_index(drop=True)

    # Convertir a float
    out["sensor_value"] = out["sensor_value"].astype(float)
    out["lag_seconds"] = out["lag_seconds"].astype(float)

    # Redondear
    out["sensor_value"] = out["sensor_value"].round(6)
    out["lag_seconds"] = out["lag_seconds"].round(6)

    return out
