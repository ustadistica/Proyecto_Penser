"""
test_train.py — Version 4.0
Tests actualizados para train.py V4 (AFE por bloques + Ward + KPrototypes)
Autores: Haider Rojas · Sergio Prieto — USTA 2026-1
"""

import sys
sys.path.insert(0, "src/percepcion")

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def df_procesado():
    """Base procesada simulada con variables de los 5 bloques AFE."""
    np.random.seed(42)
    n = 300
    # B1 Cognitivo
    b1 = ["com_escrita","com_oral","pensamiento_critico","lectura_academica",
          "argumentacion","creatividad","metodos_cualitativos","metodos_cuantitativos"]
    # B2 Tecnológico
    b2 = ["herramientas_modernas","insercion_laboral","gestion_informacion",
          "herramientas_informaticas","contextos_multiculturales",
          "conocimientos_multidisciplinares","aprendizaje_autonomo"]
    # B3 Liderazgo
    b3 = ["liderazgo","toma_decisiones","resolucion_problemas",
          "resolucion_conflictos","trabajo_equipo","investigacion","etica"]
    # B4a Satisfacción
    b4a = ["satisfaccion_formacion","efecto_calidad_vida","satisfaccion_vida"]
    # B4b Correspondencia
    b4b = ["correspondencia_primer_empleo","correspondencia_empleo_actual"]
    # B5 Bienestar
    b5 = ["adquirio_bienes","mejoro_vivienda","mejoro_salud",
          "acceso_seguridad_social","incremento_cultural","satisfecho_ocio","red_amigos"]
    # Categóricas
    cats = ["cat_genero","cat_sede","cat_estado_civil","cat_recomendaria",
            "cat_estudiaria_otra_vez","cat_nivel_educ_padres"]

    data = {}
    for c in b1+b2+b3: data[c] = np.random.choice([1,2,3,4,5], n)
    for c in b4a: data[c] = np.random.choice([0,1,2,3,4,5], n)
    for c in b4b: data[c] = np.random.choice([0,1,2,3,4,5], n)
    for c in b5: data[c] = np.random.choice([0,1], n)
    data["cat_genero"] = np.random.choice(["Masculino","Femenino"], n)
    data["cat_sede"] = np.random.choice(["Bogotá","Bucaramanga","Tunja"], n)
    data["cat_estado_civil"] = np.random.choice(["Soltero","Casado"], n)
    data["cat_recomendaria"] = np.random.choice(["Si","No"], n)
    data["cat_estudiaria_otra_vez"] = np.random.choice(["Si","No","No lo sabe"], n)
    data["cat_nivel_educ_padres"] = np.random.choice(["Media","Universitario","Basica"], n)
    data["score_bienestar"] = np.random.randint(0, 8, n)

    return pd.DataFrame(data)


# ─── Tests AFE ─────────────────────────────────────────────────────

def test_afe_bloques_retorna_dataframe(df_procesado):
    """afe_por_bloques retorna un DataFrame de scores."""
    from train import afe_por_bloques
    scores, resumen = afe_por_bloques(df_procesado)
    assert isinstance(scores, pd.DataFrame)


def test_afe_bloques_sin_nulos(df_procesado):
    """afe_por_bloques no tiene nulos en los scores."""
    from train import afe_por_bloques
    scores, _ = afe_por_bloques(df_procesado)
    assert scores.isnull().sum().sum() == 0


def test_afe_bloques_genera_indicadores(df_procesado):
    """afe_por_bloques genera al menos 5 indicadores."""
    from train import afe_por_bloques
    scores, resumen = afe_por_bloques(df_procesado)
    assert scores.shape[1] >= 5


def test_afe_bloques_conserva_registros(df_procesado):
    """afe_por_bloques conserva el número de registros."""
    from train import afe_por_bloques
    scores, _ = afe_por_bloques(df_procesado)
    assert len(scores) == len(df_procesado)


def test_afe_resumen_contiene_kmo(df_procesado):
    """El resumen AFE contiene KMO para cada bloque."""
    from train import afe_por_bloques
    _, resumen = afe_por_bloques(df_procesado)
    for nombre, info in resumen.items():
        assert "kmo" in info
        assert 0 <= info["kmo"] <= 1


# ─── Tests estandarización ─────────────────────────────────────────

def test_estandarizar_media_cero(df_procesado):
    """El espacio latente estandarizado tiene media ~0."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10


def test_estandarizar_std_uno(df_procesado):
    """El espacio latente estandarizado tiene std ~1."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)


# ─── Tests ACM ─────────────────────────────────────────────────────

def test_acm_retorna_dataframe(df_procesado):
    """aplicar_acm retorna un DataFrame."""
    from train import aplicar_acm
    coords, mca, inercia = aplicar_acm(df_procesado)
    assert isinstance(coords, pd.DataFrame)


def test_acm_sin_nulos(df_procesado):
    """aplicar_acm no tiene nulos."""
    from train import aplicar_acm
    coords, _, _ = aplicar_acm(df_procesado)
    assert coords.isnull().sum().sum() == 0


def test_acm_tres_dimensiones(df_procesado):
    """aplicar_acm retorna 3 dimensiones."""
    from train import aplicar_acm
    coords, _, _ = aplicar_acm(df_procesado)
    assert coords.shape[1] == 3


# ─── Tests clustering ──────────────────────────────────────────────

def test_clustering_ward_genera_etiquetas(df_procesado):
    """evaluar_ward genera etiquetas para todos los registros."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente, evaluar_ward
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    resultados = evaluar_ward(X_scaled, k_range=range(2, 4))
    assert 2 in resultados
    assert len(resultados[2]["labels"]) == len(df_procesado)


def test_clustering_ward_k_validos(df_procesado):
    """evaluar_ward genera k clusters correctos."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente, evaluar_ward
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    resultados = evaluar_ward(X_scaled, k_range=range(2, 5))
    for k, v in resultados.items():
        assert len(np.unique(v["labels"])) == k


def test_metricas_contienen_campos(df_procesado):
    """evaluar_ward retorna campos sil, db, dunn, bal, score."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente, evaluar_ward
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    resultados = evaluar_ward(X_scaled, k_range=range(2, 4))
    for k, v in resultados.items():
        assert "sil" in v
        assert "dunn" in v
        assert "db" in v
        assert "bal" in v
        assert "score" in v


def test_silueta_en_rango(df_procesado):
    """El coeficiente de silueta está en [-1, 1]."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente, evaluar_ward
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    resultados = evaluar_ward(X_scaled, k_range=range(2, 4))
    for k, v in resultados.items():
        assert -1 <= v["sil"] <= 1


def test_dunn_no_negativo(df_procesado):
    """El índice de Dunn es >= 0."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente, evaluar_ward
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    resultados = evaluar_ward(X_scaled, k_range=range(2, 4))
    for k, v in resultados.items():
        assert v["dunn"] >= 0


def test_seleccionar_k_retorna_dos_valores(df_procesado):
    """seleccionar_mejor_k retorna k_optimo y k_segundo distintos."""
    from train import afe_por_bloques, aplicar_acm, construir_espacio_latente, evaluar_ward, seleccionar_mejor_k
    scores, _ = afe_por_bloques(df_procesado)
    coords, _, _ = aplicar_acm(df_procesado)
    X_scaled, _, _ = construir_espacio_latente(scores, coords)
    resultados = evaluar_ward(X_scaled, k_range=range(2, 6))
    k_opt, k_sec = seleccionar_mejor_k(resultados, "Ward", k_min=2)
    assert k_opt != k_sec
    assert k_opt in resultados
    assert k_sec in resultados