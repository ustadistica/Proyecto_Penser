"""
test_features.py
================
Pruebas de sanidad para src/features.py — Proyecto PENSER USTA 2025-2
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_base():
    """DataFrame simulado que representa la salida de ingest.py."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        # Competencias (Likert 1-5)
        "          Exponer las ideas de forma clara y efectiva por medios   escritos\n        ": np.random.choice([1,2,3,4,5], n),
        "Exponer oralmente las ideas de manera clara y efectiva": np.random.choice([1,2,3,4,5], n),
        "Pensamiento crítico y analítico": np.random.choice([1,2,3,4,5], n),
        "Habilidades para trabajar en equipo": np.random.choice([1,2,3,4,5], n),
        "Capacidad de identificar problemas éticos y morales": np.random.choice([1,2,3,4,5], n),
        # Bienestar (Si/No)
        "¿Adquirió bienes materiales después de obtener su título de pregrado y posgrado?\xa0(casa, carro, inversión a largo plazo).": np.random.choice(["Si", "No"], n),
        "¿La calidad de su vivienda mejoró después de obtener su título de pregrado y posgrado?\xa0(compra o adecuaciones).": np.random.choice(["Si", "No"], n),
        "¿Continúa en contacto con su red de amigos universitarios?": np.random.choice(["Si", "No"], n),
        # Estrato
        "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado": np.random.choice([1,2,3,4,5,6], n).astype(float),
        "Estrato socioeconómico actual": np.random.choice([1,2,3,4,5,6], n).astype(float),
        # Satisfacción
        "Indique el grado de satisfacción general que tiene de su vida después de obtener su título de pregrado y posgrado": np.random.choice(["Nulo (0)", "Regular (1)", "Medio (3)", "Adecuado (4)", "Pleno (5)"], n),
        "Indique el efecto que tuvo su título de pregrado o posgrado en el mejoramiento de su calidad de vida": np.random.choice(["Medio (3)", "Adecuado (4)", "Pleno (5)"], n),
        # Sociodemográficas
        "Genero:": np.random.choice(["Masculino", "Femenino", "No binario"], n),
        "En su relación como estudiante en la Universidad Santo Tomás. ¿De qué sede o seccional es graduado?:": np.random.choice(["Bogotá", "Bucaramanga", "Tunja"], n),
        "Cuál es el mayor nivel de estudios o título universitario obtenido (responda su último nivel de estudios aprobado):": np.random.choice(["Pregrado", "Posgrado Especialización", "Posgrado Maestría"], n),
    })


# ---------------------------------------------------------------------------
# Tests de renombrado
# ---------------------------------------------------------------------------

def test_renombrar_columnas_competencias(df_base):
    """Verifica que las competencias se renombran a nombres cortos."""
    from features import renombrar_columnas
    df_out = renombrar_columnas(df_base.copy())
    assert "com_escrita" in df_out.columns
    assert "trabajo_equipo" in df_out.columns
    assert "etica" in df_out.columns


def test_renombrar_columnas_bienestar(df_base):
    """Verifica que las variables de bienestar se renombran."""
    from features import renombrar_columnas
    df_out = renombrar_columnas(df_base.copy())
    assert "adquirio_bienes" in df_out.columns
    assert "mejoro_vivienda" in df_out.columns
    assert "red_amigos" in df_out.columns


# ---------------------------------------------------------------------------
# Tests de limpieza Likert
# ---------------------------------------------------------------------------

def test_limpiar_competencias_rango_valido(df_base):
    """Verifica que las competencias quedan en rango [1-5] después de limpiar."""
    from features import renombrar_columnas, limpiar_competencias, COLS_COMPETENCIAS
    df_out = renombrar_columnas(df_base.copy())
    df_out = limpiar_competencias(df_out)
    cols = [c for c in COLS_COMPETENCIAS if c in df_out.columns]
    for col in cols:
        vals = pd.to_numeric(df_out[col], errors="coerce").dropna()
        assert vals.between(1, 5).all(), f"Valores fuera de rango en {col}"


def test_limpiar_competencias_outlier_a_nan():
    """Verifica que outliers en Likert se convierten a NaN."""
    from features import limpiar_competencias, COLS_COMPETENCIAS
    col = COLS_COMPETENCIAS[0]
    df = pd.DataFrame({col: [1.0, 3.0, 99.0, 5.0]})
    df_out = limpiar_competencias(df.copy())
    assert pd.isna(df_out[col].iloc[2])


# ---------------------------------------------------------------------------
# Tests de codificación binaria
# ---------------------------------------------------------------------------

def test_codificar_binarias_solo_cero_uno(df_base):
    """Verifica que las binarias quedan como 0 o 1 (o NaN)."""
    from features import renombrar_columnas, codificar_binarias, COLS_BIENESTAR
    df_out = renombrar_columnas(df_base.copy())
    df_out = codificar_binarias(df_out)
    cols = [c for c in COLS_BIENESTAR if c in df_out.columns]
    for col in cols:
        vals = pd.to_numeric(df_out[col], errors="coerce").dropna()
        assert vals.isin([0, 1]).all(), f"Valores inválidos en {col}"


def test_codificar_binarias_si_es_uno():
    """Verifica que 'Si' se convierte a 1."""
    from features import renombrar_columnas, codificar_binarias, COLS_BIENESTAR
    col_orig = "¿Adquirió bienes materiales después de obtener su título de pregrado y posgrado?\xa0(casa, carro, inversión a largo plazo)."
    col_nuevo = "adquirio_bienes"
    df = pd.DataFrame({col_orig: ["Si", "No", "Si", "No"]})
    df_out = renombrar_columnas(df.copy())
    df_out = codificar_binarias(df_out)
    assert df_out[col_nuevo].tolist() == [1, 0, 1, 0]


# ---------------------------------------------------------------------------
# Tests de movilidad social
# ---------------------------------------------------------------------------

def test_movilidad_social_calculo_correcto(df_base):
    """Verifica que movilidad = estrato_actual - estrato_al_graduar."""
    from features import calcular_movilidad_social
    df = pd.DataFrame({
        "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado": [2.0, 3.0, 4.0],
        "Estrato socioeconómico actual": [3.0, 3.0, 2.0],
    })
    df_out = calcular_movilidad_social(df.copy())
    assert df_out["movilidad_social"].tolist() == [1.0, 0.0, -2.0]


def test_movilidad_social_columna_creada(df_base):
    """Verifica que la columna movilidad_social existe en el output."""
    from features import calcular_movilidad_social
    df_out = calcular_movilidad_social(df_base.copy())
    assert "movilidad_social" in df_out.columns


# ---------------------------------------------------------------------------
# Tests de score de bienestar
# ---------------------------------------------------------------------------

def test_score_bienestar_rango_valido(df_base):
    """Verifica que el score de bienestar está entre 0 y 7."""
    from features import renombrar_columnas, codificar_binarias, calcular_score_bienestar
    df_out = renombrar_columnas(df_base.copy())
    df_out = codificar_binarias(df_out)
    df_out = calcular_score_bienestar(df_out)
    assert "score_bienestar" in df_out.columns
    vals = df_out["score_bienestar"].dropna()
    assert vals.between(0, 7).all()


# ---------------------------------------------------------------------------
# Tests de satisfacción
# ---------------------------------------------------------------------------

def test_codificar_satisfaccion_valores_correctos(df_base):
    """Verifica que la escala textual se convierte correctamente a numérica."""
    from features import codificar_satisfaccion
    df_out = codificar_satisfaccion(df_base.copy())
    if "satisfaccion_vida" in df_out.columns:
        vals = pd.to_numeric(df_out["satisfaccion_vida"], errors="coerce").dropna()
        assert vals.between(0, 5).all()


# ---------------------------------------------------------------------------
# Tests del pipeline completo
# ---------------------------------------------------------------------------

def test_pipeline_completo_no_falla(df_base, tmp_path):
    """Verifica que el pipeline completo de features corre sin errores."""
    from features import (
        renombrar_columnas, limpiar_competencias, codificar_binarias,
        codificar_satisfaccion, calcular_movilidad_social,
        calcular_score_bienestar
    )
    df = df_base.copy()
    df = renombrar_columnas(df)
    df = limpiar_competencias(df)
    df = codificar_binarias(df)
    df = codificar_satisfaccion(df)
    df = calcular_movilidad_social(df)
    df = calcular_score_bienestar(df)
    assert len(df) == len(df_base)
    assert "score_bienestar" in df.columns
    assert "movilidad_social" in df.columns