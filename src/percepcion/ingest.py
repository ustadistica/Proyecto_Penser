"""
ingest.py
=========
Carga, validación y diagnóstico de calidad de datos — Estudio PENSER Egresados USTA.

Autores : Yeimy Alarcón · Karen Suarez · Maria José Galindo
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2025-2

Decisiones de limpieza documentadas:
--------------------------------------
1. Columna índice oculto (Unnamed: 0): artefacto del Excel, no es variable.
2. PII eliminada: nombre y email de graduados (protección de datos).
3. Filas completamente vacías: 2 registros sin ninguna respuesta.
4. Fila fantasma: 1 registro con números de pregunta (87-109) en lugar de respuestas.
5. Columnas >95% nulos: 45 columnas de programas/ciudades con casi ningún dato.
6. Filas con >90% nulos: formularios abiertos pero no completados (~3.5 respuestas de 157).
7. Duplicados REALES: filas idénticas en columnas clave de identificación del graduado.
   Se usa subconjunto de columnas para evitar falsos duplicados por nulos.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAW_PATH     = Path("data/raw")
INTERIM_PATH = Path("data/interim")
ARTIFACTS_PATH = Path("artifacts")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
COL_INDEX_OCULTO = "Unnamed: 0"
VALOR_FILA_FANTASMA = 87.0
UMBRAL_NULOS_COLUMNAS = 0.95   # columnas con >95% nulos → eliminar
UMBRAL_NULOS_FILAS    = 0.90   # filas con >90% nulos → eliminar

# Columnas clave para detectar duplicados REALES
# (sede, programa, fecha graduación, género, estrato)
COLS_CLAVE_DUPLICADOS = [
    "En su relación como estudiante en la Universidad Santo Tomás. ¿De qué sede o seccional es graduado?:",
    "Fecha de nacimiento:",
    "Genero:",
    "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado",
    "Estrato socioeconómico actual",
    "FECHA DE GRADUACIÓN (si no esta seguro de la fecha exacta, seleccione el año)",
]

# Columnas Likert (escala 1-5)
COLS_LIKERT = [
    "          Exponer las ideas de forma clara y efectiva por medios   escritos\n        ",
    "Exponer oralmente las ideas de manera clara y efectiva",
    "Pensamiento crítico y analítico",
    "Uso y comprensión de razonamiento y métodos cuantitativos",
    "Uso y comprensión de métodos cualitativos",
    "Leer y comprender material académico (artículos, libros, etc)",
    "Capacidad argumentativa",
    "Escribir o hablar en una segunda lengua",
    "Crear ideas originales y soluciones",
    "Capacidad de resolver conflictos interpersonales",
    "Habilidades de liderazgo",
    "Asumir responsabilidades y tomar decisiones",
    "Identificar, plantear y resolver problemas",
    "Habilidad para formular, ejecutar y evaluar una investigación o proyecto",
    "Utilizar herramientas informáticas básicas",
    "Capacidad para comprender y desenvolverse en contextos multiculturales",
    "Conocimientos y habilidades relacionados con la inserción en el mercado laboral",
    "Capacidad de usar las técnicas, habilidades y herramientas modernas necesarias para la inserción en el mercado laboral",
    "Buscar, analizar, administrar y compartir información",
    "Habilidades para trabajar en equipo",
    "Capacidad de aprender y mantenerse actualizado por su cuenta",
    "Adquirir conocimientos de distintas áreas",
    "Capacidad de identificar problemas éticos y morales",
]

# Columnas binarias Si/No
COLS_BINARIAS = [
    "¿Adquirió bienes materiales después de obtener su título de pregrado y posgrado?\xa0(casa, carro, inversión a largo plazo).",
    "¿La calidad de su vivienda mejoró después de obtener su título de pregrado y posgrado?\xa0(compra o adecuaciones).",
    "¿Sus condiciones de salud han mejorado desde que obtuvo su título de pregrado o posgrado en la Universidad Santo Tomás?",
    "¿Pudo acceder a un esquema de seguridad social con mayores privilegios después de obtener su título de pregrado y posgrado?",
    "¿Su asistencia a eventos culturales, deportivos o artísticos se incrementó después de obtener su título de pregrado y posgrado?",
    "¿Está satisfecho con el tiempo de ocio que tiene disponible después de obtener su título de pregrado y posgrado?",
    "¿Continúa en contacto con su red de amigos universitarios?",
]

# PII - datos personales identificables
COLS_PII = [
    "¿Cuál es su nombre completo?:",
    "¿Cuál es su dirección de correo electrónico?:",
]


# ---------------------------------------------------------------------------
# Reporte de calidad
# ---------------------------------------------------------------------------
@dataclass
class ReporteCalidad:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    filas_originales: int = 0
    columnas_originales: int = 0
    filas_completamente_vacias: int = 0
    duplicados_reales: int = 0
    fila_fantasma: bool = False
    pii_eliminadas: list = field(default_factory=list)
    columnas_vacias_eliminadas: int = 0
    filas_incompletas_eliminadas: int = 0
    outliers_likert: dict = field(default_factory=dict)
    valores_raros_binarias: dict = field(default_factory=dict)
    estrato_anomalias: int = 0
    filas_finales: int = 0
    columnas_finales: int = 0

    def imprimir(self) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  REPORTE DE CALIDAD — ESTUDIO PENSER EGRESADOS USTA")
        print(f"  Generado: {self.timestamp}")
        print(sep)

        print(f"\n📦 DIMENSIONES")
        print(f"   Original : {self.filas_originales:,} filas × {self.columnas_originales} columnas")
        print(f"   Final    : {self.filas_finales:,} filas × {self.columnas_finales} columnas")
        eliminadas_f = self.filas_originales - self.filas_finales
        eliminadas_c = self.columnas_originales - self.columnas_finales
        print(f"   Eliminadas: {eliminadas_f} filas · {eliminadas_c} columnas")

        print(f"\n🔍 PROBLEMAS DETECTADOS")

        _ok = lambda n: f"⚠️  {n}" if n > 0 else "✅ Ninguno/a"

        print(f"   Filas completamente vacías    : {_ok(self.filas_completamente_vacias)}")
        print(f"   Duplicados reales             : {_ok(self.duplicados_reales)}")

        estado_ff = "⚠️  Detectada y eliminada" if self.fila_fantasma else "✅ No encontrada"
        print(f"   Fila fantasma (nums pregunta) : {estado_ff}")

        if self.pii_eliminadas:
            print(f"   Datos personales (PII)        : ⚠️  {len(self.pii_eliminadas)} columna(s) eliminadas")
            for c in self.pii_eliminadas:
                print(f"      → {c[:65]}")
        else:
            print(f"   Datos personales (PII)        : ✅ No encontrados")

        print(f"   Columnas >95% nulos           : {_ok(self.columnas_vacias_eliminadas)}")
        print(f"   Filas formulario incompleto   : {_ok(self.filas_incompletas_eliminadas)}")

        if self.outliers_likert:
            print(f"\n   Outliers Likert [1-5]:")
            for col, info in self.outliers_likert.items():
                print(f"      ⚠️  '{col[:50]}...' → {info['n']} valor(es): {info['valores']}")
        else:
            print(f"   Outliers Likert               : ✅ Ninguno")

        if self.valores_raros_binarias:
            print(f"\n   Valores inesperados Si/No:")
            for col, vals in self.valores_raros_binarias.items():
                print(f"      ⚠️  '{col[:50]}...' → {vals}")
        else:
            print(f"   Valores raros Si/No           : ✅ Ninguno")

        print(f"   Anomalías estrato             : {_ok(self.estrato_anomalias)}")
        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_excel(filename: str = "ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx") -> pd.DataFrame:
    ruta = RAW_PATH / filename
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {ruta}\n"
            f"Copia el Excel en la carpeta data/raw/"
        )
    log.info(f"Cargando: {ruta}")
    df = pd.read_excel(ruta)
    log.info(f"Cargado: {df.shape[0]:,} registros × {df.shape[1]} columnas")
    return df


def eliminar_columna_indice(df: pd.DataFrame) -> pd.DataFrame:
    if COL_INDEX_OCULTO in df.columns:
        df = df.drop(columns=[COL_INDEX_OCULTO])
        log.info("Columna índice oculto eliminada.")
    return df


def eliminar_pii(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    eliminadas = [c for c in COLS_PII if c in df.columns]
    if eliminadas:
        df = df.drop(columns=eliminadas)
        r.pii_eliminadas = eliminadas
        log.warning(f"PII eliminada: {len(eliminadas)} columna(s).")
    return df


def eliminar_filas_completamente_vacias(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    mask = df.isnull().all(axis=1)
    n = int(mask.sum())
    if n:
        df = df[~mask].copy()
        r.filas_completamente_vacias = n
        log.warning(f"Filas completamente vacías eliminadas: {n}.")
    else:
        log.info("Sin filas completamente vacías.")
    return df


def eliminar_duplicados(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    """
    Detecta duplicados usando solo columnas clave de identificación.
    Esto evita falsos duplicados causados por filas con muchos nulos
    que parecen iguales al comparar todas las columnas.
    """
    cols_disponibles = [c for c in COLS_CLAVE_DUPLICADOS if c in df.columns]
    if not cols_disponibles:
        log.warning("No se encontraron columnas clave para detección de duplicados.")
        return df
    n_antes = len(df)
    df = df.drop_duplicates(keep="first")
    n = n_antes - len(df)
    r.duplicados_reales = n
    if n:
        log.warning(f"Duplicados reales eliminados: {n} (filas 100% idénticas en todas las columnas).")
    else:
        log.info("Sin duplicados reales.")
    return df


def eliminar_fila_fantasma(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    col = COLS_LIKERT[0]
    if col not in df.columns:
        return df
    mask = df[col] == VALOR_FILA_FANTASMA
    if mask.any():
        df = df[~mask].copy()
        r.fila_fantasma = True
        log.warning("Fila fantasma eliminada (tenía números de pregunta en lugar de respuestas).")
    return df


def eliminar_columnas_vacias(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    pct = df.isnull().mean()
    cols = pct[pct > UMBRAL_NULOS_COLUMNAS].index.tolist()
    df = df.drop(columns=cols)
    r.columnas_vacias_eliminadas = len(cols)
    log.info(f"Columnas con >{UMBRAL_NULOS_COLUMNAS*100:.0f}% nulos eliminadas: {len(cols)}.")
    return df


def eliminar_filas_incompletas(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    """
    Elimina filas con más del 90% de nulos DESPUÉS de eliminar columnas vacías.
    Estos son formularios abiertos pero no completados
    (en promedio respondieron solo 3.5 de 157 preguntas).
    """
    mask = df.isnull().mean(axis=1) > UMBRAL_NULOS_FILAS
    n = int(mask.sum())
    if n:
        df = df[~mask].copy()
        r.filas_incompletas_eliminadas = n
        log.warning(f"Filas con formulario incompleto eliminadas: {n} (respondieron <10% del formulario).")
    else:
        log.info("Sin filas con formulario incompleto.")
    return df


def validar_likert(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    cols = [c for c in COLS_LIKERT if c in df.columns]
    for col in cols:
        serie = pd.to_numeric(df[col], errors="coerce")
        mask = serie.notna() & ~serie.between(1, 5)
        if mask.any():
            r.outliers_likert[col] = {"n": int(mask.sum()), "valores": serie[mask].unique().tolist()}
            df[col] = serie.where(~mask, other=np.nan)
    if r.outliers_likert:
        log.warning(f"Outliers Likert corregidos: {len(r.outliers_likert)} columna(s).")
    else:
        log.info("Escala Likert: sin outliers.")
    return df


def validar_binarias(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    validos = {"Si", "No", "SI", "NO", "si", "no"}
    cols = [c for c in COLS_BINARIAS if c in df.columns]
    for col in cols:
        mask = df[col].notna() & ~df[col].isin(validos)
        if mask.any():
            r.valores_raros_binarias[col] = df.loc[mask, col].unique().tolist()
            df.loc[mask, col] = np.nan
    if r.valores_raros_binarias:
        log.warning(f"Valores inválidos en binarias corregidos: {len(r.valores_raros_binarias)} columna(s).")
    else:
        log.info("Columnas Si/No: sin valores inesperados.")
    return df


def validar_estrato(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    col_grad   = "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado"
    col_actual = "Estrato socioeconómico actual"
    anomalias = 0
    for col in [col_grad, col_actual]:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce")
            mask = serie.notna() & ~serie.between(1, 6)
            if mask.any():
                anomalias += int(mask.sum())
                df[col] = serie.where(~mask, other=np.nan)
    r.estrato_anomalias = anomalias
    if anomalias:
        log.warning(f"Estrato: {anomalias} valor(es) inválido(s) → NaN.")
    else:
        log.info("Estrato socioeconómico: sin anomalías.")
    return df


def guardar_interim(df: pd.DataFrame, filename: str = "base_cargada.parquet") -> None:
    INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    # Convertir columnas object mixtas a string para compatibilidad con Parquet
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), other=None)
    ruta = INTERIM_PATH / filename
    df.to_parquet(ruta, index=False)
    log.info(f"Base guardada en: {ruta}")
    log.info(f"Dimensiones finales: {df.shape[0]:,} filas × {df.shape[1]} columnas")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(filename: str = "ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx") -> pd.DataFrame:
    r = ReporteCalidad()

    # Carga
    df = cargar_excel(filename)
    r.filas_originales    = df.shape[0]
    r.columnas_originales = df.shape[1]

    # Limpieza estructural
    df = eliminar_columna_indice(df)
    df = eliminar_pii(df, r)

    # Limpieza de registros (orden importa)
    df = eliminar_filas_completamente_vacias(df, r)
    df = eliminar_fila_fantasma(df, r)
    df = eliminar_duplicados(df, r)
    df = eliminar_columnas_vacias(df, r)
    df = eliminar_filas_incompletas(df, r)

    # Validación de variables analíticas
    df = validar_likert(df, r)
    df = validar_binarias(df, r)
    df = validar_estrato(df, r)

    # Reporte y guardado
    r.filas_finales    = df.shape[0]
    r.columnas_finales = df.shape[1]
    r.imprimir()
    guardar_interim(df)

    return df


if __name__ == "__main__":
    run()