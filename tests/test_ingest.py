"""
test_ingest.py
==============
Pruebas de sanidad para src/ingest.py — Proyecto PENSER USTA 2025-2
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_limpio():
    """Base de datos simulada limpia para pruebas."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Unnamed: 0": range(n),
        "¿Cuál es su nombre completo?:": [f"Persona {i}" for i in range(n)],
        "¿Cuál es su dirección de correo electrónico?:": [f"p{i}@mail.com" for i in range(n)],
        "Genero:": np.random.choice(["Masculino", "Femenino"], n),
        "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado": np.random.choice([1,2,3,4,5,6], n),
        "Estrato socioeconómico actual": np.random.choice([1,2,3,4,5,6], n),
        "competencia_1": np.random.choice([1,2,3,4,5], n, p=[0.05,0.1,0.2,0.4,0.25]),
        "competencia_2": np.random.choice([1,2,3,4,5], n, p=[0.05,0.1,0.2,0.4,0.25]),
    })


@pytest.fixture
def df_con_problemas():
    """Base con todos los problemas que ingest.py debe detectar."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "Unnamed: 0": range(n),
        "¿Cuál es su nombre completo?:": [f"Persona {i}" for i in range(n)],
        "¿Cuál es su dirección de correo electrónico?:": [f"p{i}@mail.com" for i in range(n)],
        "Genero:": np.random.choice(["Masculino", "Femenino"], n),
        "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado": np.random.choice([1,2,3,4,5,6], n),
        "Estrato socioeconómico actual": np.random.choice([1,2,3,4,5,6,41], n),  # 41 = error
        "col_casi_vacia": [None] * 49 + ["valor"],  # 98% nulos
        "competencia_1": list(np.random.choice([1,2,3,4,5], n-2)) + [87.0, None],  # outlier
        "binaria_1": ["Si", "No"] * 24 + ["Si", 128],  # valor raro
    })
    # Agregar duplicado exacto
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    # Agregar fila completamente vacía
    fila_vacia = pd.DataFrame([{col: None for col in df.columns}])
    df = pd.concat([df, fila_vacia], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Tests de eliminación de columna índice
# ---------------------------------------------------------------------------

def test_eliminar_columna_indice_existe(df_limpio):
    """Verifica que se elimina Unnamed: 0 cuando existe."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_columna_indice
    from ingest import ReporteCalidad

    assert "Unnamed: 0" in df_limpio.columns
    df_out = eliminar_columna_indice(df_limpio.copy())
    assert "Unnamed: 0" not in df_out.columns


def test_eliminar_columna_indice_no_existe():
    """Verifica que no falla si Unnamed: 0 no existe."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_columna_indice

    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    df_out = eliminar_columna_indice(df.copy())
    assert df_out.shape == df.shape


# ---------------------------------------------------------------------------
# Tests de eliminación de PII
# ---------------------------------------------------------------------------

def test_eliminar_pii_remueve_columnas(df_limpio):
    """Verifica que nombre y email son eliminados."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_pii, ReporteCalidad

    r = ReporteCalidad()
    df_out = eliminar_pii(df_limpio.copy(), r)
    assert "¿Cuál es su nombre completo?:" not in df_out.columns
    assert "¿Cuál es su dirección de correo electrónico?:" not in df_out.columns


def test_eliminar_pii_registra_en_reporte(df_limpio):
    """Verifica que el reporte registra las columnas PII eliminadas."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_pii, ReporteCalidad

    r = ReporteCalidad()
    eliminar_pii(df_limpio.copy(), r)
    assert len(r.pii_eliminadas) == 2


# ---------------------------------------------------------------------------
# Tests de duplicados
# ---------------------------------------------------------------------------

def test_eliminar_duplicados_detecta_exactos():
    """Verifica que se eliminan filas 100% idénticas usando columnas clave PENSER."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_duplicados, ReporteCalidad

    df = pd.DataFrame({
        "En su relación como estudiante en la Universidad Santo Tomás. ¿De qué sede o seccional es graduado?:": ["Bogotá", "Bucaramanga", "Bogotá"],
        "Fecha de nacimiento:": ["1990-01-01", "1995-05-05", "1990-01-01"],
        "Genero:": ["Masculino", "Femenino", "Masculino"],
        "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado": [3, 4, 3],
        "Estrato socioeconómico actual": [4, 4, 4],
        "FECHA DE GRADUACIÓN (si no esta seguro de la fecha exacta, seleccione el año)": ["2020", "2019", "2020"],
        "col_extra": ["a", "b", "a"],
    })
    r = ReporteCalidad()
    df_out = eliminar_duplicados(df.copy(), r)
    assert len(df_out) == 2
    assert r.duplicados_reales == 1


def test_eliminar_duplicados_sin_duplicados():
    """Verifica que no elimina nada cuando no hay duplicados."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_duplicados, ReporteCalidad

    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": ["x", "y", "z"],
    })
    r = ReporteCalidad()
    df_out = eliminar_duplicados(df.copy(), r)
    assert len(df_out) == 3
    assert r.duplicados_reales == 0


# ---------------------------------------------------------------------------
# Tests de filas completamente vacías
# ---------------------------------------------------------------------------

def test_eliminar_filas_vacias():
    """Verifica que se eliminan filas con todos los valores nulos."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_filas_completamente_vacias, ReporteCalidad

    df = pd.DataFrame({
        "col_a": [1, None, 3],
        "col_b": ["x", None, "z"],
    })
    r = ReporteCalidad()
    df_out = eliminar_filas_completamente_vacias(df.copy(), r)
    assert len(df_out) == 2
    assert r.filas_completamente_vacias == 1


# ---------------------------------------------------------------------------
# Tests de validación Likert
# ---------------------------------------------------------------------------

def test_validar_likert_corrige_outliers():
    """Verifica que valores fuera de [1-5] se convierten a NaN."""
    import sys
    sys.path.insert(0, "src")
    from ingest import validar_likert, ReporteCalidad, COLS_LIKERT

    col = COLS_LIKERT[0]
    df = pd.DataFrame({col: [1.0, 3.0, 87.0, 5.0, None]})
    r = ReporteCalidad()
    df_out = validar_likert(df.copy(), r)
    assert pd.isna(df_out[col].iloc[2])
    assert len(r.outliers_likert) == 1


def test_validar_likert_sin_outliers():
    """Verifica que valores válidos [1-5] no se modifican."""
    import sys
    sys.path.insert(0, "src")
    from ingest import validar_likert, ReporteCalidad, COLS_LIKERT

    col = COLS_LIKERT[0]
    df = pd.DataFrame({col: [1.0, 2.0, 3.0, 4.0, 5.0]})
    r = ReporteCalidad()
    df_out = validar_likert(df.copy(), r)
    assert len(r.outliers_likert) == 0
    assert df_out[col].notna().all()


# ---------------------------------------------------------------------------
# Tests de validación binarias
# ---------------------------------------------------------------------------

def test_validar_binarias_corrige_valores_raros():
    """Verifica que valores distintos de Si/No se convierten a NaN."""
    import sys
    sys.path.insert(0, "src")
    from ingest import validar_binarias, ReporteCalidad, COLS_BINARIAS

    col = COLS_BINARIAS[0]
    df = pd.DataFrame({col: ["Si", "No", 128, "Si"]})
    r = ReporteCalidad()
    df_out = validar_binarias(df.copy(), r)
    assert pd.isna(df_out[col].iloc[2])
    assert len(r.valores_raros_binarias) == 1


# ---------------------------------------------------------------------------
# Tests de validación estrato
# ---------------------------------------------------------------------------

def test_validar_estrato_corrige_anomalias():
    """Verifica que valores de estrato fuera de [1-6] se convierten a NaN."""
    import sys
    sys.path.insert(0, "src")
    from ingest import validar_estrato, ReporteCalidad

    col = "Estrato socioeconómico actual"
    df = pd.DataFrame({col: [1.0, 3.0, 41.0, 6.0]})
    r = ReporteCalidad()
    df_out = validar_estrato(df.copy(), r)
    assert pd.isna(df_out[col].iloc[2])
    assert r.estrato_anomalias == 1


def test_validar_estrato_valido_sin_cambios():
    """Verifica que estratos válidos [1-6] no se modifican."""
    import sys
    sys.path.insert(0, "src")
    from ingest import validar_estrato, ReporteCalidad

    col = "Estrato socioeconómico actual"
    df = pd.DataFrame({col: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    r = ReporteCalidad()
    df_out = validar_estrato(df.copy(), r)
    assert r.estrato_anomalias == 0
    assert df_out[col].notna().all()


# ---------------------------------------------------------------------------
# Tests de columnas casi vacías
# ---------------------------------------------------------------------------

def test_eliminar_columnas_vacias():
    """Verifica que columnas con >95% nulos se eliminan."""
    import sys
    sys.path.insert(0, "src")
    from ingest import eliminar_columnas_vacias, ReporteCalidad

    df = pd.DataFrame({
        "col_buena": [1, 2, 3, 4, 5],
        "col_vacia": [None, None, None, None, 1],  # 80% nulos
        "col_casi_vacia": [None] * 99 + [1] if False else [None, None, None, None, None],  # 100% nulos
    })
    df["col_casi_vacia"] = None
    r = ReporteCalidad()
    df_out = eliminar_columnas_vacias(df.copy(), r)
    assert "col_casi_vacia" not in df_out.columns
    assert r.columnas_vacias_eliminadas >= 1