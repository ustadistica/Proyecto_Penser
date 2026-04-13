"""
test_train.py
=============
Pruebas de sanidad para src/train.py — Proyecto PENSER USTA 2025-2
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
def df_procesado():
    """DataFrame simulado que representa la salida de features.py."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "com_escrita":                    np.random.choice([1,2,3,4,5], n).astype(float),
        "com_oral":                       np.random.choice([1,2,3,4,5], n).astype(float),
        "pensamiento_critico":            np.random.choice([1,2,3,4,5], n).astype(float),
        "metodos_cuantitativos":          np.random.choice([1,2,3,4,5], n).astype(float),
        "metodos_cualitativos":           np.random.choice([1,2,3,4,5], n).astype(float),
        "lectura_academica":              np.random.choice([1,2,3,4,5], n).astype(float),
        "argumentacion":                  np.random.choice([1,2,3,4,5], n).astype(float),
        "segunda_lengua":                 np.random.choice([1,2,3,4,5], n).astype(float),
        "creatividad":                    np.random.choice([1,2,3,4,5], n).astype(float),
        "resolucion_conflictos":          np.random.choice([1,2,3,4,5], n).astype(float),
        "liderazgo":                      np.random.choice([1,2,3,4,5], n).astype(float),
        "toma_decisiones":                np.random.choice([1,2,3,4,5], n).astype(float),
        "resolucion_problemas":           np.random.choice([1,2,3,4,5], n).astype(float),
        "investigacion":                  np.random.choice([1,2,3,4,5], n).astype(float),
        "herramientas_informaticas":      np.random.choice([1,2,3,4,5], n).astype(float),
        "contextos_multiculturales":      np.random.choice([1,2,3,4,5], n).astype(float),
        "insercion_laboral":              np.random.choice([1,2,3,4,5], n).astype(float),
        "herramientas_modernas":          np.random.choice([1,2,3,4,5], n).astype(float),
        "gestion_informacion":            np.random.choice([1,2,3,4,5], n).astype(float),
        "trabajo_equipo":                 np.random.choice([1,2,3,4,5], n).astype(float),
        "aprendizaje_autonomo":           np.random.choice([1,2,3,4,5], n).astype(float),
        "conocimientos_multidisciplinares": np.random.choice([1,2,3,4,5], n).astype(float),
        "etica":                          np.random.choice([1,2,3,4,5], n).astype(float),
        "adquirio_bienes":                np.random.choice([0,1], n).astype(float),
        "mejoro_vivienda":                np.random.choice([0,1], n).astype(float),
        "mejoro_salud":                   np.random.choice([0,1], n).astype(float),
        "acceso_seguridad_social":        np.random.choice([0,1], n).astype(float),
        "incremento_cultural":            np.random.choice([0,1], n).astype(float),
        "satisfecho_ocio":                np.random.choice([0,1], n).astype(float),
        "red_amigos":                     np.random.choice([0,1], n).astype(float),
        "movilidad_social":               np.random.choice([-2,-1,0,1,2], n).astype(float),
        "satisfaccion_vida":              np.random.choice([1,2,3,4,5], n).astype(float),
        "satisfaccion_formacion":         np.random.choice([1,2,3,4,5], n).astype(float),
        "efecto_calidad_vida":            np.random.choice([1,2,3,4,5], n).astype(float),
        "score_bienestar":                np.random.choice(range(8), n).astype(float),
    })


# ---------------------------------------------------------------------------
# Tests de selección de variables
# ---------------------------------------------------------------------------

def test_seleccionar_variables_retorna_dataframe(df_procesado):
    """Verifica que seleccionar_variables retorna un DataFrame."""
    from train import seleccionar_variables
    X = seleccionar_variables(df_procesado.copy())
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(df_procesado)


def test_seleccionar_variables_solo_analiticas(df_procesado):
    """Verifica que solo incluye variables analíticas definidas."""
    from train import seleccionar_variables, COLS_COMPETENCIAS, COLS_BIENESTAR, COLS_ADICIONALES
    X = seleccionar_variables(df_procesado.copy())
    todas = set(COLS_COMPETENCIAS + COLS_BIENESTAR + COLS_ADICIONALES)
    for col in X.columns:
        assert col in todas


# ---------------------------------------------------------------------------
# Tests de imputación
# ---------------------------------------------------------------------------

def test_imputar_nulos_elimina_nulos(df_procesado):
    """Verifica que no quedan nulos después de imputar."""
    from train import seleccionar_variables, imputar_nulos
    X = seleccionar_variables(df_procesado.copy())
    # Insertar algunos nulos
    X.iloc[0, 0] = np.nan
    X.iloc[5, 3] = np.nan
    X_imp = imputar_nulos(X)
    assert X_imp.isnull().sum().sum() == 0


def test_imputar_nulos_conserva_dimensiones(df_procesado):
    """Verifica que la imputación no cambia el número de filas ni columnas."""
    from train import seleccionar_variables, imputar_nulos
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    assert X_imp.shape == X.shape


# ---------------------------------------------------------------------------
# Tests de estandarización
# ---------------------------------------------------------------------------

def test_estandarizar_media_cero(df_procesado):
    """Verifica que la media de cada columna estandarizada es ~0."""
    from train import seleccionar_variables, imputar_nulos, estandarizar
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    medias = np.abs(X_scaled.mean(axis=0))
    assert (medias < 1e-10).all(), "Medias no son ~0 después de estandarizar"


def test_estandarizar_std_uno(df_procesado):
    """Verifica que la desviación estándar de cada columna estandarizada es ~1."""
    from train import seleccionar_variables, imputar_nulos, estandarizar
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    stds = np.abs(X_scaled.std(axis=0))
    assert (np.abs(stds - 1) < 1e-10).all(), "STDs no son ~1 después de estandarizar"


# ---------------------------------------------------------------------------
# Tests de PCA
# ---------------------------------------------------------------------------

def test_pca_reduce_dimensiones(df_procesado):
    """Verifica que PCA reduce el número de columnas."""
    from train import seleccionar_variables, imputar_nulos, estandarizar, aplicar_pca
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    X_pca, pca = aplicar_pca(X_scaled, varianza_objetivo=0.85)
    assert X_pca.shape[1] < X_scaled.shape[1]
    assert X_pca.shape[0] == X_scaled.shape[0]


def test_pca_varianza_objetivo_cumplida(df_procesado):
    """Verifica que los componentes PCA explican al menos el 85% de varianza."""
    from train import seleccionar_variables, imputar_nulos, estandarizar, aplicar_pca
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    X_pca, pca = aplicar_pca(X_scaled, varianza_objetivo=0.85)
    varianza_total = sum(pca.explained_variance_ratio_) * 100
    assert varianza_total >= 85.0


# ---------------------------------------------------------------------------
# Tests de clustering
# ---------------------------------------------------------------------------

def test_clustering_genera_etiquetas_validas(df_procesado):
    """Verifica que el clustering genera etiquetas para todos los registros."""
    from train import seleccionar_variables, imputar_nulos, estandarizar, aplicar_pca, entrenar_modelo_final
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    X_pca, _ = aplicar_pca(X_scaled)
    labels, _ = entrenar_modelo_final(X_pca, "kmeans", k=3)
    assert len(labels) == len(df_procesado)
    assert set(labels).issubset({0, 1, 2})


def test_clustering_jerarquico_genera_etiquetas(df_procesado):
    """Verifica que el clustering jerárquico también funciona."""
    from train import seleccionar_variables, imputar_nulos, estandarizar, aplicar_pca, entrenar_modelo_final
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    X_pca, _ = aplicar_pca(X_scaled)
    labels, _ = entrenar_modelo_final(X_pca, "jerarquico", k=3)
    assert len(labels) == len(df_procesado)
    assert set(labels).issubset({0, 1, 2})


def test_evaluacion_clustering_retorna_metricas(df_procesado):
    """Verifica que la evaluación retorna métricas para cada k."""
    from train import seleccionar_variables, imputar_nulos, estandarizar, aplicar_pca, evaluar_clustering
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    X_pca, _ = aplicar_pca(X_scaled)
    resultados = evaluar_clustering(X_pca, k_range=range(2, 5))
    assert "kmeans" in resultados
    assert "jerarquico" in resultados
    for k in range(2, 5):
        assert k in resultados["kmeans"]
        assert "silueta" in resultados["kmeans"][k]
        assert "davies_bouldin" in resultados["kmeans"][k]


def test_silueta_entre_menos1_y_1(df_procesado):
    """Verifica que el coeficiente de silueta está en rango válido [-1, 1]."""
    from train import seleccionar_variables, imputar_nulos, estandarizar, aplicar_pca, evaluar_clustering
    X = seleccionar_variables(df_procesado.copy())
    X_imp = imputar_nulos(X)
    X_scaled, _ = estandarizar(X_imp)
    X_pca, _ = aplicar_pca(X_scaled)
    resultados = evaluar_clustering(X_pca, k_range=range(2, 4))
    for metodo in resultados:
        for k, metricas in resultados[metodo].items():
            assert -1 <= metricas["silueta"] <= 1