"""
features_depurada.py
====================
Depuración, codificación y construcción de variables analíticas
— Base Depurada Estudio PENSER USTA 2025.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Decisiones de codificación documentadas:
------------------------------------------
1. Escala logro (Muy bajo→Muy alto) → ordinal 1-5
2. "No incidió" en incidencia → 0
3. Binarias Si/No → 1/0
4. Tipo de cargo → ordinal 1-5
5. Tipo de vinculación → ordinal 1-5
6. Score de impacto: suma de 6 variables de incidencia (0-30)
7. Score de competencias: suma de 15 variables de logro (0-75)
"""

import logging
import warnings
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

INTERIM_PATH   = Path("data/interim")
PROCESSED_PATH = Path("data/processed")

ESCALA_LOGRO = {"Muy bajo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5}
ESCALA_INCIDENCIA = {"No incidió": 0, "Muy bajo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5}

ESCALA_TIPO_CARGO = {"Operativo": 1, "Asistencial": 2, "Técnico": 3, "Profesional": 4, "Directivo": 5}
ESCALA_VINCULACION = {
    "Contrato de aprendizaje": 1,
    "Contrato de prestación de servicios (por obra o labor)": 2,
    "Propietario de empresa": 3,
    "Contrato laboral a término fijo": 4,
    "Contrato laboral a término indefinido": 5,
}
ESCALA_PERCEPCION = {
    "Totalmente en desacuerdo": 1, "En desacuerdo": 2,
    "Ni de acuerdo ni en desacuerdo": 3, "De acuerdo": 4, "Totalmente de acuerdo": 5,
}
ESCALA_SUFICIENCIA = {
    "Muy insuficientes": 1, "Insuficientes": 2,
    "Ni suficientes ni insuficientes": 3, "Suficientes": 4, "Muy suficientes": 5,
}
ESCALA_NIVEL_ESTUDIOS = {
    "Bachiller": 1, "Técnico": 2, "Tecnólogo": 3, "Profesional": 4,
    "Especialista": 5, "Magister": 6, "Doctor": 7,
}
ESCALA_INGRESO = {
    "Muy insuficiente (mis ingresos no me alcanzan para cubrir las necesidades básicas de mi hogar)": 1,
    "Insuficiente (mis ingresos no son suficientes para cubrir todas mis necesidades básicas)": 2,
    "Suficiente (mis ingresos cubren satisfactoriamente mis necesidades básicas, aunque no me permiten tener dinero adicional para usarlo como desee)": 3,
    "Muy suficiente (mis ingresos cubren mis necesidades básicas y me permiten tener dinero adicional para usarlo como desee)": 4,
}

RENAME_COMP_TRANS = {'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [COMUNICACIÓN EFECTIVA: expresar con claridad, y en forma apropiada al contexto y la cultura, lo q': 'logro_comunicacion', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [RELACIONES INTERPERSONALES: establecer y conservar relaciones significativas, así como ser capaz ': 'logro_relaciones_interpersonales', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [TOMA DE DECISIONES: evaluar distintas alternativas, teniendo en cuenta necesidades, capacidades, ': 'logro_toma_decisiones', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [SOLUCIÓN DE PROBLEMAS Y CONFLICTOS: transformar y manejar los problemas y conflictos de la vida d': 'logro_solucion_problemas', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [PENSAMIENTO CREATIVO: usar la razón y la “pasión” (emociones, sentimientos, intuición, fantasías ': 'logro_pensamiento_creativo', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [PENSAMIENTO CRÍTICO: aprender a preguntarse, investigar y no aceptar las cosas de forma crédula. ': 'logro_pensamiento_critico', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [MANEJO DE EMOCIONES Y SENTIMIENTOS: aprender a navegar en el mundo afectivo logrando mayor “sinto': 'logro_manejo_emociones', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [MANEJO DE TENSIONES Y ESTRÉS: identificar oportunamente las fuentes de tensión y estrés en la vid': 'logro_manejo_estres', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [MULTICULTURALES: tener conocimiento u comprensión de distintas culturas.]': 'logro_multiculturales', 'II 9. A partir de la siguiente escala indique el nivel en que la Universidad favoreció en usted el desarrollo de las siguientes competencias transversales.  [INTERCULTURALES: buenas actitudes que se mantienen hacia otras culturas.]': 'logro_interculturales'}

RENAME_COMP_PROF = {'Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, regio': 'logro_comp_prof_1', 'Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re2': 'logro_comp_prof_2', 'Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re3': 'logro_comp_prof_3', 'Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re4': 'logro_comp_prof_4', 'Indique el nivel de logro en el que la formación recibida durante sus estudios favoreció el desarrollo de las competencias profesionales, requeridas en los diferentes campos de su desempeño y pertinentes con las necesidades de los contextos locales, re5': 'logro_comp_prof_5'}

RENAME_INCIDENCIA = {'IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El mejoramiento de mis ingresos]': 'incidencia_ingresos', 'IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a oportunidades de empleo]': 'incidencia_empleo', 'IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El mejoramiento de las condiciones de la vivienda en términos de infraestructura o ubicación (por ejemplo, se trasladó a una mejor zona de la ciudad)]': 'incidencia_vivienda', 'IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a servicios de salud]': 'incidencia_salud', 'IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a servicios de recreación y deporte]': 'incidencia_recreacion', 'IV. 21. Indique el nivel de incidencia de la formación recibida en:  [El acceso a mayores niveles educativos]': 'incidencia_educacion'}

RENAME_BINARIAS = {'I.1. ¿Se encuentra laborando actualmente? ': 'laborando_actualmente', 'I.5. ¿Ha recibido distinciones o premios? ': 'recibio_distinciones', 'I.6.  ¿Usted pertenece a algún gremio, red y/o asociación científica? ': 'pertenece_gremio', 'I.7. Una vez graduado (a) del programa, ¿ha liderado o acompañado proyectos comunitarios?': 'lidera_proyectos_comunitarios', 'I.8. Una vez graduado (a) del programa, ¿ha liderado o participado en proyectos de investigación?': 'lidera_investigacion', 'III.18. ¿El valor actual de sus ingresos mensuales es superior al valor de los ingresos mensuales durante su último año de estudio? ': 'ingresos_superiores_estudio'}


COLS_LOGRO_TRANS = list(RENAME_COMP_TRANS.values())
COLS_LOGRO_PROF  = list(RENAME_COMP_PROF.values())
COLS_LOGRO       = COLS_LOGRO_PROF + COLS_LOGRO_TRANS
COLS_INCIDENCIA  = list(RENAME_INCIDENCIA.values())
COLS_BINARIAS    = list(RENAME_BINARIAS.values())


def renombrar_columnas(df):
    rename_map = {**RENAME_COMP_PROF, **RENAME_COMP_TRANS, **RENAME_INCIDENCIA, **RENAME_BINARIAS}
    df = df.rename(columns=rename_map)
    log.info(f"Columnas renombradas: {len(rename_map)} "
             f"({len(RENAME_COMP_PROF)} comp.prof + {len(RENAME_COMP_TRANS)} comp.trans + "
             f"{len(RENAME_INCIDENCIA)} incidencia + {len(RENAME_BINARIAS)} binarias)")
    return df


def codificar_logro(df):
    for col in COLS_LOGRO:
        if col in df.columns:
            df[col] = df[col].map(ESCALA_LOGRO)
    for col in COLS_INCIDENCIA:
        if col in df.columns:
            df[col] = df[col].map(ESCALA_INCIDENCIA)
    log.info(f"Escala logro codificada: {len(COLS_LOGRO)} comp + {len(COLS_INCIDENCIA)} incidencia.")
    return df


def codificar_binarias(df):
    for col in COLS_BINARIAS:
        if col in df.columns:
            df[col] = df[col].map({"Si": 1, "No": 0, "SI": 1, "NO": 0})
    log.info(f"Variables binarias codificadas: {len(COLS_BINARIAS)}")
    return df


def codificar_trayectoria(df):
    col_cargo = [c for c in df.columns if "Tipo de cargo" in c or "tipo de cargo" in c]
    if col_cargo:
        df["tipo_cargo"] = df[col_cargo[0]].map(ESCALA_TIPO_CARGO)
    col_vinc = [c for c in df.columns if "Tipo de vinculación" in c or "tipo de vinculación" in c]
    if col_vinc:
        df["tipo_vinculacion"] = df[col_vinc[0]].map(ESCALA_VINCULACION)
    log.info("Trayectoria laboral codificada.")
    return df


def codificar_percepcion_programa(df):
    col = [c for c in df.columns if c.startswith("I.4. Indique su percepción")]
    if col:
        df["percepcion_programa"] = df[col[0]].map(ESCALA_PERCEPCION)
    col2 = [c for c in df.columns if c.startswith("I.4.I.")]
    if col2:
        df["competencias_suficientes"] = df[col2[0]].map(ESCALA_SUFICIENCIA)
    log.info("Percepción del programa codificada.")
    return df


def codificar_nivel_estudios(df):
    col = [c for c in df.columns if c.startswith("III.13.")]
    if col:
        df["nivel_estudios"] = df[col[0]].map(ESCALA_NIVEL_ESTUDIOS)
    log.info("Nivel de estudios codificado.")
    return df


def codificar_ingreso(df):
    col = [c for c in df.columns if c.startswith("III.19.")]
    if col:
        df["percepcion_ingreso"] = df[col[0]].map(ESCALA_INGRESO)
    log.info("Percepción de ingreso codificada.")
    return df


def codificar_formacion_impacto(df):
    col = [c for c in df.columns if c.startswith("IV.20.")]
    if col:
        df["formacion_impacto_general"] = df[col[0]].map(ESCALA_PERCEPCION)
    log.info("Impacto general de la formación codificado.")
    return df


def codificar_categoricas_acm(df):
    col_sede = [c for c in df.columns if c.startswith("Sede o Seccional")]
    if col_sede:
        df["cat_sede"] = df[col_sede[0]].str.strip()

    col_prog = [c for c in df.columns if c.startswith("PROGRAMA ACADEMICO")]
    if col_prog:
        df["cat_programa"] = df[col_prog[0]].str.strip().str.upper()
        freq = df["cat_programa"].value_counts()
        df["cat_programa"] = df["cat_programa"].apply(
            lambda x: x if pd.isna(x) or freq.get(x, 0) >= 10 else "Otro"
        )

    col_cargo = [c for c in df.columns if "Tipo de cargo" in c]
    if col_cargo:
        df["cat_tipo_cargo"] = df[col_cargo[0]].str.strip()

    if "laborando_actualmente" in df.columns:
        df["cat_laborando"] = df["laborando_actualmente"].map({1: "Si", 0: "No"})

    log.info("Variables categóricas para ACM preparadas.")
    return df


def calcular_scores(df):
    cols_trans = [c for c in COLS_LOGRO_TRANS if c in df.columns]
    if cols_trans:
        df["score_logro_competencias"] = df[cols_trans].sum(axis=1, skipna=True)
        log.info(f"Score logro competencias: media={df['score_logro_competencias'].mean():.2f} / {len(cols_trans)*5}")

    cols_inc = [c for c in COLS_INCIDENCIA if c in df.columns]
    if cols_inc:
        df["score_impacto_formacion"] = df[cols_inc].sum(axis=1, skipna=True)
        log.info(f"Score impacto formación: media={df['score_impacto_formacion'].mean():.2f} / {len(cols_inc)*5}")

    cols_act = ["recibio_distinciones", "pertenece_gremio",
                "lidera_proyectos_comunitarios", "lidera_investigacion"]
    cols_ok = [c for c in cols_act if c in df.columns]
    if cols_ok:
        df["score_actividad_profesional"] = df[cols_ok].sum(axis=1, skipna=True)
        log.info(f"Score actividad profesional: media={df['score_actividad_profesional'].mean():.2f} / {len(cols_ok)}")
    return df


def resumen_variables(df):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  RESUMEN DE VARIABLES — FEATURES_DEPURADA.PY")
    print(sep)
    comp  = [c for c in df.columns if c.startswith("logro_")]
    inc   = [c for c in COLS_INCIDENCIA if c in df.columns]
    bin_  = [c for c in COLS_BINARIAS if c in df.columns]
    cats  = [c for c in df.columns if c.startswith("cat_")]
    ords  = ["tipo_cargo", "tipo_vinculacion", "percepcion_programa",
             "competencias_suficientes", "nivel_estudios",
             "percepcion_ingreso", "formacion_impacto_general"]
    ords_ok = [c for c in ords if c in df.columns]
    print(f"\n📊 VARIABLES ANALÍTICAS")
    print(f"   Logro competencias (1-5)    : {len(comp)}")
    print(f"   Incidencia formación (0-5)  : {len(inc)}")
    print(f"   Binarias 0/1                : {len(bin_)}")
    print(f"   Ordinales construidas       : {len(ords_ok)}")
    print(f"   Categóricas para ACM        : {len(cats)}")
    print(f"   Total columnas              : {df.shape[1]}")
    print(f"   Total registros             : {df.shape[0]:,}")
    print(f"\n📈 ESTADÍSTICAS LOGRO COMPETENCIAS")
    if comp:
        stats = df[comp].describe().loc[["mean","std","min","max"]].round(2)
        print(stats.T.to_string())
    print(f"\n📈 ESTADÍSTICAS INCIDENCIA FORMACIÓN")
    if inc:
        stats = df[inc].describe().loc[["mean","std","min","max"]].round(2)
        print(stats.T.to_string())
    print(f"\n✅ SCORES COMPUESTOS")
    for sc in ["score_logro_competencias", "score_impacto_formacion", "score_actividad_profesional"]:
        if sc in df.columns:
            print(f"   {sc}: media={df[sc].mean():.2f}, max={df[sc].max():.0f}")
    print(f"\n🏷️  CATEGÓRICAS PARA ACM")
    for c in cats:
        print(f"   {c:35} | categorías={df[c].nunique()} | nulos={df[c].isnull().mean()*100:.1f}%")
    print(f"\n{sep}\n")


def guardar_procesada(df, filename="base_depurada_procesada.parquet"):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), other=None)
    ruta = PROCESSED_PATH / filename
    df.to_parquet(ruta, index=False)
    log.info(f"Base procesada guardada en: {ruta}")


def run():
    log.info("Iniciando features_depurada.py...")
    df = pd.read_parquet(INTERIM_PATH / "base_depurada_cargada.parquet")
    log.info(f"Base cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    df = renombrar_columnas(df)
    df = codificar_logro(df)
    df = codificar_binarias(df)
    df = codificar_trayectoria(df)
    df = codificar_percepcion_programa(df)
    df = codificar_nivel_estudios(df)
    df = codificar_ingreso(df)
    df = codificar_formacion_impacto(df)
    df = codificar_categoricas_acm(df)
    df = calcular_scores(df)
    resumen_variables(df)
    guardar_procesada(df)
    return df


if __name__ == "__main__":
    run()