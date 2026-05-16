import pandas as pd
import numpy as np

def analizar_series_por_grupo(
    datos: pd.DataFrame,
    columna_grupo: str,
    columna_tiempo: str,
    columna_valor: str,
    ventana: int,
    umbral_zscore: float
) -> pd.DataFrame:
    
    # Copia para no modificar el DataFrame original
    df = datos.copy()

    # Guardar índice original
    df["_original_index"] = df.index

    # Ordenar por grupo y tiempo
    df = df.sort_values(by=[columna_grupo, columna_tiempo])

    # Media móvil por grupo
    df["media_movil"] = (
        df.groupby(columna_grupo)[columna_valor]
        .transform(
            lambda x: x.rolling(
                window=ventana,
                min_periods=ventana
            ).mean()
        )
    )

    # Desviación estándar móvil por grupo
    df["std_movil"] = (
        df.groupby(columna_grupo)[columna_valor]
        .transform(
            lambda x: x.rolling(
                window=ventana,
                min_periods=ventana
            ).std()
        )
    )

    # Z-score
    df["zscore"] = np.where(
        df["std_movil"] > 0,
        (df[columna_valor] - df["media_movil"]) / df["std_movil"],
        np.nan
    )

    # Detección de cambio
    df["es_cambio"] = df["zscore"].abs() > umbral_zscore

    # Donde zscore sea NaN -> False
    df["es_cambio"] = df["es_cambio"].where(
        df["zscore"].notna(),
        other=False
    )

    # Restaurar orden original
    df = (
        df.sort_values("_original_index")
          .drop(columns=["_original_index"])
    )

    return df
