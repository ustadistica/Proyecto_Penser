"""
features.py
===========
Depuración, codificación y construcción de variables analíticas — Estudio PENSER.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Dimensiones construidas:
-------------------------
1. Competencias (23 vars Likert 1-5)     → renombradas a nombres cortos
2. Bienestar                             → 15 binarias codificadas 0/1
3. Satisfacción con la formación         → 6 escalas textuales → numéricas
4. Trayectoria laboral                   → ordinales codificadas
5. Movilidad social                      → estrato_actual - estrato_al_graduar
6. Variables categóricas nominales       → limpiadas para ACM en train.py
7. Nivel de cargo                        → recodificado de texto libre a 6 niveles

Decisiones documentadas:
-------------------------
- Los valores numéricos raros (39, 40, 119, 125, 128-156, etc.) son artefactos
  de exportación del formulario (números de pregunta). Se convierten a NaN.
- nivel_cargo tiene 119 categorías en texto libre. Se recodifica a 6 niveles
  estándar usando coincidencia de palabras clave.
- Las variables categóricas nominales se mantienen como string limpio para
  que train.py pueda aplicar ACM directamente sobre ellas.
"""

import logging
import re
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

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
INTERIM_PATH   = Path("data/interim")
PROCESSED_PATH = Path("data/processed")

# ---------------------------------------------------------------------------
# Mapeo competencias → nombres cortos
# ---------------------------------------------------------------------------
RENAME_COMPETENCIAS = {
    "          Exponer las ideas de forma clara y efectiva por medios   escritos\n        ": "com_escrita",
    "Exponer oralmente las ideas de manera clara y efectiva":                                "com_oral",
    "Pensamiento crítico y analítico":                                                       "pensamiento_critico",
    "Uso y comprensión de razonamiento y métodos cuantitativos":                             "metodos_cuantitativos",
    "Uso y comprensión de métodos cualitativos":                                             "metodos_cualitativos",
    "Leer y comprender material académico (artículos, libros, etc)":                        "lectura_academica",
    "Capacidad argumentativa":                                                               "argumentacion",
    "Escribir o hablar en una segunda lengua":                                               "segunda_lengua",
    "Crear ideas originales y soluciones":                                                   "creatividad",
    "Capacidad de resolver conflictos interpersonales":                                      "resolucion_conflictos",
    "Habilidades de liderazgo":                                                              "liderazgo",
    "Asumir responsabilidades y tomar decisiones":                                           "toma_decisiones",
    "Identificar, plantear y resolver problemas":                                            "resolucion_problemas",
    "Habilidad para formular, ejecutar y evaluar una investigación o proyecto":              "investigacion",
    "Utilizar herramientas informáticas básicas":                                            "herramientas_informaticas",
    "Capacidad para comprender y desenvolverse en contextos multiculturales":                "contextos_multiculturales",
    "Conocimientos y habilidades relacionados con la inserción en el mercado laboral":       "insercion_laboral",
    "Capacidad de usar las técnicas, habilidades y herramientas modernas necesarias para la inserción en el mercado laboral": "herramientas_modernas",
    "Buscar, analizar, administrar y compartir información":                                 "gestion_informacion",
    "Habilidades para trabajar en equipo":                                                   "trabajo_equipo",
    "Capacidad de aprender y mantenerse actualizado por su cuenta":                         "aprendizaje_autonomo",
    "Adquirir conocimientos de distintas áreas":                                             "conocimientos_multidisciplinares",
    "Capacidad de identificar problemas éticos y morales":                                   "etica",
}

COLS_COMPETENCIAS = list(RENAME_COMPETENCIAS.values())

# ---------------------------------------------------------------------------
# Mapeo bienestar → nombres cortos (7 variables de impacto post-grado)
# ---------------------------------------------------------------------------
RENAME_BIENESTAR = {
    "¿Adquirió bienes materiales después de obtener su título de pregrado y posgrado?\xa0(casa, carro, inversión a largo plazo).": "adquirio_bienes",
    "¿La calidad de su vivienda mejoró después de obtener su título de pregrado y posgrado?\xa0(compra o adecuaciones).":          "mejoro_vivienda",
    "¿Sus condiciones de salud han mejorado desde que obtuvo su título de pregrado o posgrado en la Universidad Santo Tomás?":     "mejoro_salud",
    "¿Pudo acceder a un esquema de seguridad social con mayores privilegios después de obtener su título de pregrado y posgrado?": "acceso_seguridad_social",
    "¿Su asistencia a eventos culturales, deportivos o artísticos se incrementó después de obtener su título de pregrado y posgrado?": "incremento_cultural",
    "¿Está satisfecho con el tiempo de ocio que tiene disponible después de obtener su título de pregrado y posgrado?":            "satisfecho_ocio",
    "¿Continúa en contacto con su red de amigos universitarios?":                                                                  "red_amigos",
}

COLS_BIENESTAR = list(RENAME_BIENESTAR.values())

# ---------------------------------------------------------------------------
# Mapeo bienestar adicional (variables binarias de contexto familiar/laboral)
# ---------------------------------------------------------------------------
RENAME_BINARIAS_ADICIONALES = {
    "¿Realizó una práctica o pasantía en una organización externa a la Universidad Santo Tomás durante sus estudios?": "realizo_practica",
    "¿Ha recomendado a amigos, familiares o conocidos el programa del que es graduado?":                                "recomendaria_usta",
    "¿En la actualidad continúa ejerciendo su primer empleo?":                                                          "continua_primer_empleo",
}

# ---------------------------------------------------------------------------
# Escalas de satisfacción (texto → número 0-5)
# ---------------------------------------------------------------------------
ESCALA_SATISFACCION = {
    "Nulo (0)": 0, "Regular (1)": 1, "Aceptable (2)": 2,
    "Medio (3)": 3, "Adecuado (4)": 4, "Pleno (5)": 5,
}

# ---------------------------------------------------------------------------
# Escalas ordinales de trayectoria laboral
# ---------------------------------------------------------------------------
ESCALA_TIEMPO_EMPLEO = {
    "Estaba vinculado laboralmente antes de obtener el título": 0,
    "Menos de 1 mes": 1,
    "Entre 1 mes y 3 meses": 2,
    "Entre 3 meses y 6 meses": 3,
    "Entre 6 meses y 1 año": 4,
    "Entre 1 año y 2 años": 5,
    "Más de 1 año": 5,
    "Más de 2 años": 6,
}

ESCALA_RELACION_SECTOR = {
    "Nada relacionado": 1,
    "Relación Indirecta": 2,
    "Relación Directa": 3,
}

ESCALA_SALARIO_PREGRADO = {
    "Menor a\xa0$ 2.100.000": 1,
    "Entre\xa0$2.100.000 y\xa0 $2.500.000": 2,
    "Entre $2.500.000 y  $3.500.000": 3,
    "Mayor a $3.500.000": 4,
}

ESCALA_SALARIO_ESPECIALIZACION = {
    "Menor a\xa0$ 3.000.000": 1,
    "Entre\xa0$3.000.000 y\xa0 $3.500.000": 2,
    "Entre $3.500.000 y  $4.500.000": 3,
    "Mayor a $4.500.000": 4,
}

ESCALA_SALARIO_MAESTRIA = {
    "Menor a\xa0$ 4.000.000": 1,
    "Entre\xa0$4.000.000 y\xa0 $4.500.000": 2,
    "Entre $4.500.000 y  $5.000.000": 3,
    "Mayor a $5.000.000": 4,
}

# ---------------------------------------------------------------------------
# Recodificación nivel de cargo (texto libre → 6 niveles estándar)
# ---------------------------------------------------------------------------
# Reglas de coincidencia por palabras clave (orden importa — más específico primero)
NIVEL_CARGO_REGLAS = [
    # Nivel 0 — Sin empleo
    (0, ["no tengo empleo", "sin empleo", "sin trabajo", "no he conseguido",
         "aún no he conseguido", "no tengo trabajo", "no tengo", "ninguno",
         "na", "n/a"]),
    # Nivel 1 — Operativo / Auxiliar
    (1, ["auxiliar", "operativo", "asistente", "secretar", "cajera", "vendedor",
         "vendedora", "citador", "escribiente", "becaria", "practicante",
         "niñera", "empleada", "apoyo", "atención al usuario", "call center",
         "agente"]),
    # Nivel 2 — Técnico / Analista junior
    (2, ["técnico", "tecnico", "analista", "junior", "dev", "investigador",
         "litigante", "abogada", "contratista", "independiente", "autonomo",
         "prestacion de servicios", "comerciante", "preparador"]),
    # Nivel 3 — Profesional
    (3, ["profesional", "profesora", "docente", "interventor", "consultor",
         "asesor", "ejecutiva", "apoyo profesional"]),
    # Nivel 4 — Coordinador / Jefe
    (4, ["coordinador", "coordinadora", "jefe", "supervisor", "directora del grado"]),
    # Nivel 5 — Directivo / Gerente
    (5, ["directivo", "directora", "director", "gerente", "manager",
         "senior", "autonomo, ordenador"]),
]


def _recodificar_cargo(valor: str) -> float:
    """Asigna nivel estándar a una descripción de cargo en texto libre."""
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().lower()
    # Valores numéricos raros (números de pregunta del formulario)
    if v.isdigit():
        return np.nan
    for nivel, palabras_clave in NIVEL_CARGO_REGLAS:
        if any(kw in v for kw in palabras_clave):
            return float(nivel)
    # Default: profesional si no coincide con nada
    return 3.0


# ---------------------------------------------------------------------------
# Columnas categóricas nominales para ACM
# (se limpian aquí, se usan en train.py)
# ---------------------------------------------------------------------------
COLS_CATEGORICAS_ACM = {
    "Genero:": "cat_genero",
    "En su relación como estudiante en la Universidad Santo Tomás. ¿De qué sede o seccional es graduado?:": "cat_sede",
    "Estado civil:": "cat_estado_civil",
    "Tipo de contrato:2": "cat_tipo_contrato",
    "¿Ha recomendado a amigos, familiares o conocidos el programa del que es graduado?": "cat_recomendaria",
    "¿Estudiaría otro programa de pregrado o posgrado en la Universidad Santo Tomás?": "cat_estudiaria_otra_vez",
}

# Valores válidos por columna categórica (el resto → NaN)
VALS_VALIDOS_CAT = {
    "cat_genero":           {"Masculino", "Femenino", "No binario"},
    "cat_sede":             {"Bogotá", "Bucaramanga", "Villavicencio", "Tunja",
                             "Medellín", "Educación abierta y a distancia"},
    "cat_estado_civil":     {"Soltero", "Casado", "Unión libre", "Separado",
                             "Viudo", "Vida Religiosa"},
    "cat_tipo_contrato":    {"A término indefinido", "A término fijo",
                             "Por prestación de servicios", "Otro",
                             "Contrato de aprendizaje"},
    "cat_recomendaria":     {"Si", "No"},
    "cat_estudiaria_otra_vez": {"Si", "No", "No lo sabe"},
}

# Nivel educativo padres → recodificado a 5 categorías comparables
NIVEL_EDUC_PADRES_MAP = {
    "Ninguno": "Sin_estudios",
    "Primaria": "Basica",
    "Bachiller": "Media",
    "Técnico": "Tecnico",
    "Tecnólogo": "Tecnico",
    "Tecnólogo ": "Tecnico",
    "Profesional": "Universitario",
    "Especialización": "Posgrado",
    "Maestría": "Posgrado",
    "Doctorado": "Posgrado",
    "Posgrado": "Posgrado",
}


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra competencias y bienestar a nombres cortos manejables."""
    df = df.rename(columns={**RENAME_COMPETENCIAS, **RENAME_BIENESTAR})
    log.info(f"Columnas renombradas: {len(RENAME_COMPETENCIAS)} competencias + {len(RENAME_BIENESTAR)} bienestar.")
    return df


def limpiar_competencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que las competencias estén en rango [1-5].
    Valores fuera de rango (incluyendo números de pregunta residuales) → NaN.
    """
    cols = [c for c in COLS_COMPETENCIAS if c in df.columns]
    corregidas = 0
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = df[col].notna() & ~df[col].between(1, 5)
        if mask.any():
            df.loc[mask, col] = np.nan
            corregidas += 1
    log.info(f"Competencias validadas: {len(cols)} columnas. Corregidas: {corregidas}.")
    return df


def codificar_binarias_bienestar(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas bienestar Si/No a 1/0."""
    cols = [c for c in COLS_BIENESTAR if c in df.columns]
    for col in cols:
        df[col] = df[col].map({"Si": 1, "No": 0, "SI": 1, "NO": 0, "si": 1, "no": 0})
    log.info(f"Variables bienestar codificadas: {len(cols)} columnas.")
    return df


def codificar_binarias_adicionales(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica binarias adicionales de contexto (práctica, recomendaría, etc.)."""
    for col_orig, col_nuevo in RENAME_BINARIAS_ADICIONALES.items():
        col_real = [c for c in df.columns if c.startswith(col_orig[:50])]
        if col_real:
            df[col_nuevo] = df[col_real[0]].map(
                {"Si": 1, "No": 0, "SI": 1, "NO": 0, "si": 1, "no": 0}
            )
    log.info("Variables binarias adicionales codificadas.")
    return df


def codificar_satisfaccion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte escalas textuales de satisfacción a numéricas (0–5).
    Valores numéricos raros (números de pregunta) → NaN.
    """
    cols_satisfaccion = {
        "Indique el grado de cumplimiento de sus expectativas de formación con la oferta del programa.": "satisfaccion_formacion",
        "Indique el efecto que tuvo su título de pregrado o posgrado en el mejoramiento de su calidad de vida": "efecto_calidad_vida",
        "Indique el grado de satisfacción general que tiene de su vida después de obtener su título de pregrado y posgrado": "satisfaccion_vida",
        "Indique el grado de correspondencia entre sus funciones en su primer empleo y las competencias desarrolladas durante el programa de pregrado o posgrado:": "correspondencia_primer_empleo",
        "Indique el grado de correspondencia entre sus funciones en su empleo actual y las competencias desarrolladas durante el programa de pregrado o posgrado": "correspondencia_empleo_actual",
        "Indique la relación entre el sector de su empleo actual y su título de pregrado o posgrado": "relacion_sector_actual_sat",
    }
    creadas = 0
    for col_orig, col_nuevo in cols_satisfaccion.items():
        col_real = [c for c in df.columns if c.startswith(col_orig[:55])]
        if col_real:
            df[col_nuevo] = df[col_real[0]].map(ESCALA_SATISFACCION)
            creadas += 1
    log.info(f"Variables de satisfacción codificadas: {creadas} columnas.")
    return df


def calcular_movilidad_social(df: pd.DataFrame) -> pd.DataFrame:
    """
    Índice de movilidad social = estrato_actual − estrato_al_graduarse.
    Positivo = ascenso · Negativo = descenso · 0 = sin cambio.
    """
    col_grad   = "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado"
    col_actual = "Estrato socioeconómico actual"
    col_g = [c for c in df.columns if c.startswith(col_grad[:50])]
    col_a = [c for c in df.columns if c.startswith(col_actual[:30])]
    if col_g and col_a:
        e_grad   = pd.to_numeric(df[col_g[0]], errors="coerce")
        e_actual = pd.to_numeric(df[col_a[0]], errors="coerce")
        # Estrato válido: 1-6
        e_grad   = e_grad.where(e_grad.between(1, 6))
        e_actual = e_actual.where(e_actual.between(1, 6))
        df["movilidad_social"] = e_actual - e_grad
        df["estrato_grad"]    = e_grad
        df["estrato_actual"]  = e_actual
        dist = df["movilidad_social"].value_counts().sort_index()
        log.info(f"Movilidad social calculada. Distribución: {dist.to_dict()}")
    else:
        log.warning("Columnas de estrato no encontradas.")
    return df


def recodificar_nivel_cargo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recodifica nivel de cargo de texto libre a escala ordinal 0–5.
    0 = sin empleo · 1 = operativo · 2 = técnico/analista ·
    3 = profesional · 4 = coordinador · 5 = directivo/gerente
    """
    for col_prefix, col_nuevo in [
        ("Especifique el nivel del cargo desempeñado en su primer empleo:", "nivel_cargo_primer_empleo"),
        ("Especifique el nivel del cargo desempeñado en su empleo actual",  "nivel_cargo_actual"),
    ]:
        col_real = [c for c in df.columns if c.startswith(col_prefix[:45])]
        if col_real:
            df[col_nuevo] = df[col_real[0]].apply(_recodificar_cargo)
            # Excluir nivel 0 (sin empleo) del análisis — no es un nivel de cargo
            df[col_nuevo] = df[col_nuevo].where(df[col_nuevo] > 0)
    log.info("Nivel de cargo recodificado (0=sin empleo excluido, 1–5 niveles).")
    return df


def codificar_tiempo_primer_empleo(df: pd.DataFrame) -> pd.DataFrame:
    """Tiempo en conseguir primer empleo → ordinal 0–6."""
    col_prefix = "Indique el tiempo que tardó en lograr su primer empleo"
    col_real = [c for c in df.columns if c.startswith(col_prefix[:45])]
    if col_real:
        df["tiempo_primer_empleo"] = df[col_real[0]].map(ESCALA_TIEMPO_EMPLEO)
    log.info("Tiempo primer empleo codificado.")
    return df


def codificar_relacion_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Relación sector-título primer empleo → ordinal 1–3."""
    col_prefix = "Indique la relación entre el sector de su primer empleo"
    col_real = [c for c in df.columns if c.startswith(col_prefix[:45])]
    if col_real:
        df["relacion_sector_primer_empleo"] = df[col_real[0]].map(ESCALA_RELACION_SECTOR)
    log.info("Relación sector primer empleo codificada.")
    return df


def codificar_salario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica rangos salariales a escala ordinal 1–4.
    Usa el salario más alto disponible (maestría > especialización > pregrado).
    """
    # Pregrado primer empleo
    col = [c for c in df.columns if "primer trabajo" in c and "pregrado" in c.lower()]
    if col:
        df["salario_primer_pregrado"] = df[col[0]].map(ESCALA_SALARIO_PREGRADO)

    # Salario empleo actual (el más reciente disponible)
    col_pg = [c for c in df.columns if "empleo más reciente" in c and "pregrado" in c.lower()]
    col_esp = [c for c in df.columns if "empleo más reciente" in c and "Especialización" in c]
    col_mae = [c for c in df.columns if "empleo más reciente" in c and "Maestría" in c]

    salario_actual = pd.Series(np.nan, index=df.index)
    if col_pg:
        salario_actual = salario_actual.fillna(df[col_pg[0]].map(ESCALA_SALARIO_PREGRADO))
    if col_esp:
        salario_actual = salario_actual.fillna(df[col_esp[0]].map(ESCALA_SALARIO_ESPECIALIZACION))
    if col_mae:
        salario_actual = salario_actual.fillna(df[col_mae[0]].map(ESCALA_SALARIO_MAESTRIA))
    df["salario_actual"] = salario_actual
    log.info("Salarios codificados (ordinal 1–4).")
    return df


def codificar_nivel_formacion(df: pd.DataFrame) -> pd.DataFrame:
    """Nivel de formación → ordinal 1–4."""
    col_prefix = "Cuál es el mayor nivel de estudios"
    col_real = [c for c in df.columns if c.startswith(col_prefix[:30])]
    nivel_map = {
        "Pregrado": 1, "Posgrado Especialización": 2,
        "Posgrado Maestría": 3, "Posgrado Doctorado": 4,
    }
    if col_real:
        df["nivel_formacion"] = df[col_real[0]].map(nivel_map)
    log.info("Nivel de formación codificado.")
    return df


def limpiar_categoricas_acm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza las variables categóricas nominales para ACM.
    - Elimina valores numéricos raros (números de pregunta del formulario)
    - Normaliza espacios y capitalización
    - Recodifica nivel educativo de padres a 5 categorías
    """
    # Variables categóricas principales
    for col_orig, col_nuevo in COLS_CATEGORICAS_ACM.items():
        col_real = [c for c in df.columns if c.startswith(col_orig[:50])]
        if col_real:
            serie = df[col_real[0]].copy()
            # Eliminar valores numéricos raros
            serie = serie.apply(
                lambda x: np.nan if pd.isna(x) or str(x).strip().isdigit() else str(x).strip()
            )
            # Normalizar espacios en estado civil
            if col_nuevo == "cat_estado_civil":
                serie = serie.str.strip()
                serie = serie.replace({
                    "Soltero ": "Soltero", "Casado ": "Casado",
                    "Unión libre ": "Unión libre",
                    "Vida Religiosa": "Religioso",
                    "Religioso": "Religioso",
                    "sacerdote S.D.S": "Religioso",
                })
            # Mantener solo valores válidos
            if col_nuevo in VALS_VALIDOS_CAT:
                validos = VALS_VALIDOS_CAT[col_nuevo]
                serie = serie.where(serie.isin(validos))
            df[col_nuevo] = serie

    # Nivel educativo de los padres → 5 categorías
    col_nep = [c for c in df.columns if c.startswith("Nivel educativo de los padres")]
    if col_nep:
        serie = df[col_nep[0]].copy()
        serie = serie.apply(
            lambda x: np.nan if pd.isna(x) or str(x).strip().isdigit() else str(x).strip()
        )
        df["cat_nivel_educ_padres"] = serie.map(NIVEL_EDUC_PADRES_MAP)
        log.info("Nivel educativo padres recodificado a 5 categorías.")

    n_limpias = len(COLS_CATEGORICAS_ACM) + 1
    log.info(f"Variables categóricas para ACM limpias: {n_limpias} columnas.")
    return df


def calcular_score_bienestar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score compuesto de bienestar = suma de los 7 indicadores binarios.
    Rango: 0–7. Refleja acumulación de logros materiales y sociales post-grado.
    """
    cols = [c for c in COLS_BIENESTAR if c in df.columns]
    if cols:
        df["score_bienestar"] = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        log.info(f"Score bienestar calculado (0–{len(cols)}). Media: {df['score_bienestar'].mean():.2f}.")
    return df


def resumen_variables_finales(df: pd.DataFrame) -> None:
    """Imprime resumen de todas las variables construidas."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  RESUMEN DE VARIABLES CONSTRUIDAS — FEATURES.PY")
    print(sep)

    comp      = [c for c in COLS_COMPETENCIAS if c in df.columns]
    bien      = [c for c in COLS_BIENESTAR if c in df.columns]
    cats_acm  = [c for c in df.columns if c.startswith("cat_")]
    ordinales = ["movilidad_social", "nivel_cargo_actual", "nivel_cargo_primer_empleo",
                 "tiempo_primer_empleo", "relacion_sector_primer_empleo", "salario_actual",
                 "salario_primer_pregrado", "satisfaccion_formacion", "efecto_calidad_vida",
                 "satisfaccion_vida", "nivel_formacion", "score_bienestar",
                 "estrato_grad", "estrato_actual"]
    ordinales_ok = [c for c in ordinales if c in df.columns]

    print(f"\n📊 VARIABLES ANALÍTICAS")
    print(f"   Competencias Likert 1-5       : {len(comp)}")
    print(f"   Bienestar binarias 0/1        : {len(bien)}")
    print(f"   Ordinales construidas         : {len(ordinales_ok)}")
    print(f"   Categóricas para ACM          : {len(cats_acm)}")
    print(f"   Total columnas en base        : {df.shape[1]}")
    print(f"   Total registros               : {df.shape[0]:,}")

    print(f"\n📈 ESTADÍSTICAS COMPETENCIAS (escala 1–5)")
    if comp:
        stats = df[comp].apply(pd.to_numeric, errors="coerce").describe().loc[["mean","std","min","max"]].round(2)
        print(stats.T.to_string())

    print(f"\n🏠 MOVILIDAD SOCIAL")
    if "movilidad_social" in df.columns:
        mov = df["movilidad_social"].value_counts().sort_index()
        pct = (mov / mov.sum() * 100).round(1)
        for v, n in mov.items():
            label = "sin cambio" if v == 0 else ("ascenso" if v > 0 else "descenso")
            print(f"   {int(v):+d} estrato ({label}): {n} graduados ({pct[v]}%)")

    print(f"\n✅ BIENESTAR")
    if "score_bienestar" in df.columns:
        sc = df["score_bienestar"]
        print(f"   Score promedio : {sc.mean():.2f} / 7")
        print(f"   Score máximo   : {sc.max():.0f} / 7")
        print(f"   Score mínimo   : {sc.min():.0f} / 7")

    print(f"\n🏷️  VARIABLES CATEGÓRICAS PARA ACM")
    for c in cats_acm:
        n_vals = df[c].nunique()
        nulos_pct = df[c].isnull().mean() * 100
        print(f"   {c:30} | categorías={n_vals} | nulos={nulos_pct:.1f}%")

    print(f"\n{sep}\n")


def guardar_procesada(df: pd.DataFrame, filename: str = "base_procesada.parquet") -> None:
    """Guarda la base procesada en Parquet."""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), other=None)
    ruta = PROCESSED_PATH / filename
    df.to_parquet(ruta, index=False)
    log.info(f"Base procesada guardada en: {ruta}")


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Pipeline completo de construcción de variables analíticas."""
    log.info("Iniciando features.py...")
    df = pd.read_parquet(INTERIM_PATH / "base_cargada.parquet")
    log.info(f"Base cargada desde interim: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # 1. Renombrar
    df = renombrar_columnas(df)

    # 2. Limpiar variables cuantitativas
    df = limpiar_competencias(df)
    df = codificar_binarias_bienestar(df)
    df = codificar_binarias_adicionales(df)

    # 3. Construir variables ordinales
    df = codificar_satisfaccion(df)
    df = calcular_movilidad_social(df)
    df = recodificar_nivel_cargo(df)
    df = codificar_tiempo_primer_empleo(df)
    df = codificar_relacion_sector(df)
    df = codificar_salario(df)
    df = codificar_nivel_formacion(df)

    # 4. Limpiar variables categóricas para ACM
    df = limpiar_categoricas_acm(df)

    # 5. Score compuesto
    df = calcular_score_bienestar(df)

    # 6. Resumen y guardado
    resumen_variables_finales(df)
    guardar_procesada(df)

    return df


if __name__ == "__main__":
    run()