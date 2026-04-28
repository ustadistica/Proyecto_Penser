"""
ingest_depurada.py
==================
Carga, validación y diagnóstico de calidad de datos
— Base Depurada Estudio PENSER USTA 2025.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Características de esta base:
-------------------------------
- 1.129 registros × 143 columnas
- Escala de logro: Muy bajo / Bajo / Medio / Alto / Muy alto / No aplica
- 21 variables de logro (competencias + competencias transversales + incidencia)
- 6 variables binarias Si/No
- Sin duplicados exactos ni de cédula
- 89 columnas con >90% nulos (preguntas por sede específica) → eliminadas
- Sin PII sensible (tiene cédula pero no nombre ni email)

Diferencias vs base de percepción:
-------------------------------------
- Escala ordinal textual (no numérica 1-5) → requiere codificación
- "No aplica" se trata como NaN (el graduado no tuvo esa experiencia)
- "No incidió" en variables de incidencia → valor 0 (sin efecto)
- Menor tamaño muestral (1.129 vs 2.530)
- No tiene variables de bienestar Si/No equivalentes a la base de percepción
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
FILENAME_DEFAULT = "DATA_DEPURADA_PENSER_2025.xlsx"
UMBRAL_NULOS_COLUMNAS = 0.90  # columnas con >90% nulos → eliminar

# Columnas con datos personales (cédula se conserva para posible cruce futuro)
COLS_PII = [
    "Consentimiento de tratamiento de datos personales\nDe acuerdo con la política de tratamiento de datos personales, se informa que la información consignada en el presente formulario será usada únicamente con fines académicos e investigativos, salvaguardando",
]

# Columnas de logro (escala Muy bajo → Muy alto)
COLS_LOGRO_COMPETENCIAS = [
    "Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, regio",
    "Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re2",
    "Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re3",
    "Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re4",
    "Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re5",
]

COLS_LOGRO_TRANSVERSALES = [
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [COMUNICACIÓN EFECTIVA: expresar con claridad, y en forma apropiada al contexto y la cultura, lo q",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [RELACIONES INTERPERSONALES: establecer y conservar relaciones significativas, así como ser capaz ",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [TOMA DE DECISIONES: evaluar distintas alternativas, teniendo en cuenta necesidades, capacidades, ",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [SOLUCIÓN DE PROBLEMAS Y CONFLICTOS: transformar y manejar los problemas y conflictos de la vida d",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [PENSAMIENTO CREATIVO: usar la razón y la \"pasión\" (emociones, sentimientos, intuición, fantasías ",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [PENSAMIENTO CRÍTICO: aprender a preguntarse, investigar y no aceptar las cosas de forma crédula. ",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [MANEJO DE EMOCIONES Y SENTIMIENTOS: aprender a navegar en el mundo afectivo logrando mayor \"sinto",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [MANEJO DE TENSIONES Y ESTRÉS: identificar oportunamente las fuentes de tensión y estrés en la vid",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [MULTICULTURALES: tener conocimiento u comprensión de distintas culturas.]",
    "II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [INTERCULTURALES: buenas actitudes que se mantienen hacia otras culturas.]",
]

COLS_LOGRO_INCIDENCIA = [
    "IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El mejoramiento de mis ingresos]",
    "IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a oportunidades de empleo]",
    "IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El mejoramiento de las condiciones de la vivienda en términos de infraestructura o ubicación (por ejemplo, se trasladó a una mejor zona de la ciudad)]",
    "IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a servicios de salud]",
    "IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a servicios de recreación y deporte]",
    "IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a mayores niveles educativos]",
]

COLS_BINARIAS = [
    "I.1. ¿Se encuentra laborando actualmente? ",
    "I.5. ¿Ha recibido distinciones o premios? ",
    "I.6.  ¿Usted pertenece a algún gremio, red y/o asociación científica? ",
    "I.7. Una vez graduado (a) del programa, ¿ha liderado o acompañado proyectos comunitarios?",
    "I.8. Una vez graduado (a) del programa, ¿ha liderado o participado en proyectos de investigación?",
    "III.18. ¿El valor actual de sus ingresos mensuales es superior al valor de los ingresos mensuales durante su último año de estudio? ",
]

TODAS_COLS_LOGRO = COLS_LOGRO_COMPETENCIAS + COLS_LOGRO_TRANSVERSALES + COLS_LOGRO_INCIDENCIA


# ---------------------------------------------------------------------------
# Reporte de calidad
# ---------------------------------------------------------------------------
@dataclass
class ReporteCalidad:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    filas_originales: int = 0
    columnas_originales: int = 0
    duplicados_exactos: int = 0
    duplicados_cedula: int = 0
    pii_eliminadas: list = field(default_factory=list)
    columnas_vacias_eliminadas: int = 0
    valores_raros_logro: dict = field(default_factory=dict)
    no_aplica_por_columna: dict = field(default_factory=dict)
    filas_finales: int = 0
    columnas_finales: int = 0

    def imprimir(self) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  REPORTE DE CALIDAD — BASE DEPURADA PENSER USTA 2025")
        print(f"  Generado: {self.timestamp}")
        print(sep)

        print(f"\n📦 DIMENSIONES")
        print(f"   Original : {self.filas_originales:,} filas × {self.columnas_originales} columnas")
        print(f"   Final    : {self.filas_finales:,} filas × {self.columnas_finales} columnas")
        print(f"   Eliminadas: {self.filas_originales - self.filas_finales} filas · {self.columnas_originales - self.columnas_finales} columnas")

        print(f"\n🔍 PROBLEMAS DETECTADOS")
        _ok = lambda n: f"⚠️  {n}" if n > 0 else "✅ Ninguno/a"

        print(f"   Duplicados exactos     : {_ok(self.duplicados_exactos)}")
        print(f"   Duplicados por cédula  : {_ok(self.duplicados_cedula)}")

        if self.pii_eliminadas:
            print(f"   PII eliminada          : ⚠️  {len(self.pii_eliminadas)} columna(s)")
        else:
            print(f"   PII eliminada          : ✅ No encontrada")

        print(f"   Columnas >90% nulos    : ⚠️  {self.columnas_vacias_eliminadas} eliminadas")

        if self.valores_raros_logro:
            print(f"\n   Valores raros en escala logro:")
            for col, vals in self.valores_raros_logro.items():
                print(f"      ⚠️  '{col[:50]}...' → {vals}")
        else:
            print(f"   Valores raros logro    : ✅ Ninguno")

        print(f"\n   'No aplica' por columna de logro (→ NaN):")
        for col, n in self.no_aplica_por_columna.items():
            pct = n / self.filas_originales * 100
            print(f"      {col[:50]}: {n} ({pct:.1f}%)")

        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_excel(filename: str = FILENAME_DEFAULT) -> pd.DataFrame:
    ruta = RAW_PATH / filename
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {ruta}\n"
            f"Copia el Excel en data/raw/ con el nombre: {filename}"
        )
    log.info(f"Cargando: {ruta}")
    df = pd.read_excel(ruta)
    log.info(f"Cargado: {df.shape[0]:,} registros × {df.shape[1]} columnas")
    return df


def eliminar_pii(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    """Elimina columna de consentimiento (texto legal, no es variable analítica)."""
    eliminadas = [c for c in COLS_PII if c in df.columns]
    if eliminadas:
        df = df.drop(columns=eliminadas)
        r.pii_eliminadas = eliminadas
        log.warning(f"Columnas PII/consentimiento eliminadas: {len(eliminadas)}")
    return df


def verificar_duplicados(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    """Verifica duplicados exactos y por número de cédula."""
    n_exactos = int(df.duplicated().sum())
    r.duplicados_exactos = n_exactos
    if n_exactos:
        df = df.drop_duplicates()
        log.warning(f"Duplicados exactos eliminados: {n_exactos}")
    else:
        log.info("Sin duplicados exactos.")

    col_cedula = "Número de documento de identificación"
    if col_cedula in df.columns:
        n_ced = int(df[col_cedula].duplicated().sum())
        r.duplicados_cedula = n_ced
        if n_ced:
            log.warning(f"Cédulas duplicadas: {n_ced} (se conserva primera ocurrencia)")
            df = df.drop_duplicates(subset=[col_cedula], keep="first")
        else:
            log.info("Sin cédulas duplicadas.")
    return df


def eliminar_columnas_vacias(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    """Elimina columnas con >90% nulos (preguntas por sede específica)."""
    pct = df.isnull().mean()
    cols = pct[pct > UMBRAL_NULOS_COLUMNAS].index.tolist()
    df = df.drop(columns=cols)
    r.columnas_vacias_eliminadas = len(cols)
    log.info(f"Columnas con >{UMBRAL_NULOS_COLUMNAS*100:.0f}% nulos eliminadas: {len(cols)}")
    return df


def validar_escala_logro(df: pd.DataFrame, r: ReporteCalidad) -> pd.DataFrame:
    """
    Valida la escala ordinal de logro.
    Valores válidos: Muy bajo, Bajo, Medio, Alto, Muy alto, No aplica, No incidió.
    - 'No aplica' → NaN (el graduado no tuvo esa experiencia)
    - 'No incidió' → se trata como 'Muy bajo' solo en columnas de incidencia
    - Valores fuera del conjunto → NaN
    """
    vals_validos_logro = {"Muy bajo", "Bajo", "Medio", "Alto", "Muy alto", "No aplica"}
    vals_validos_incidencia = {"Muy bajo", "Bajo", "Medio", "Alto", "Muy alto",
                                "No aplica", "No incidió"}

    for col in TODAS_COLS_LOGRO:
        if col not in df.columns:
            continue

        es_incidencia = col in COLS_LOGRO_INCIDENCIA
        validos = vals_validos_incidencia if es_incidencia else vals_validos_logro

        # Detectar valores raros
        raros = df[col][~df[col].isin(validos) & df[col].notna()]
        if len(raros) > 0:
            r.valores_raros_logro[col] = raros.unique().tolist()
            df.loc[~df[col].isin(validos), col] = np.nan

        # Contar No aplica
        n_no_aplica = (df[col] == "No aplica").sum()
        if n_no_aplica > 0:
            r.no_aplica_por_columna[col[:50]] = int(n_no_aplica)
            df.loc[df[col] == "No aplica", col] = np.nan

    if r.valores_raros_logro:
        log.warning(f"Valores raros en escala logro corregidos: {len(r.valores_raros_logro)} columna(s)")
    else:
        log.info("Escala logro: sin valores raros.")

    log.info(f"'No aplica' convertido a NaN en {len(r.no_aplica_por_columna)} columna(s).")
    return df


def validar_binarias(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que las binarias solo tengan Si/No."""
    validos = {"Si", "No", "SI", "NO"}
    for col in COLS_BINARIAS:
        if col not in df.columns:
            continue
        raros = df[col][~df[col].isin(validos) & df[col].notna()]
        if len(raros) > 0:
            log.warning(f"Valores raros en '{col[:50]}': {raros.unique()} → NaN")
            df.loc[~df[col].isin(validos), col] = np.nan
    log.info("Variables binarias validadas.")
    return df


def guardar_interim(df: pd.DataFrame,
                    filename: str = "base_depurada_cargada.parquet") -> None:
    """Guarda la base validada en data/interim/."""
    INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), other=None)
    ruta = INTERIM_PATH / filename
    df.to_parquet(ruta, index=False)
    log.info(f"Base validada guardada en: {ruta}")
    log.info(f"Dimensiones finales: {df.shape[0]:,} filas × {df.shape[1]} columnas")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(filename: str = FILENAME_DEFAULT) -> pd.DataFrame:
    r = ReporteCalidad()

    df = cargar_excel(filename)
    r.filas_originales    = df.shape[0]
    r.columnas_originales = df.shape[1]

    df = eliminar_pii(df, r)
    df = verificar_duplicados(df, r)
    df = eliminar_columnas_vacias(df, r)
    df = validar_escala_logro(df, r)
    df = validar_binarias(df)

    r.filas_finales    = df.shape[0]
    r.columnas_finales = df.shape[1]
    r.imprimir()
    guardar_interim(df)

    return df


if __name__ == "__main__":
    run()