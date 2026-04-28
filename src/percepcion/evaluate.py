"""
evaluate.py
===========
Descripción, interpretación y reporte de los arquetipos de graduados USTA.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Arquetipos identificados (3) mediante AFE + ACM + Ward:
---------------------------------------------------------
0 — El Subjetivamente Satisfecho  : 594  (23.5%) — menor bienestar y satisfacción
1 — El Profesional Consolidado    : 602  (23.8%) — perfil medio equilibrado
2 — El Líder de Alto Desempeño    : 1334 (52.7%) — mayor bienestar y competencias

Factores AFE que sustentan los arquetipos:
-------------------------------------------
F1 — Competencias cognitivas y comunicativas
F2 — Satisfacción y correspondencia laboral
F3 — Competencias tecnológicas e inserción laboral
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

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ARTIFACTS_PATH = Path("artifacts")

# ---------------------------------------------------------------------------
# Definición de arquetipos
# ---------------------------------------------------------------------------
ARQUETIPOS = {
    0: {
        "nombre": "El Subjetivamente Satisfecho",
        "descripcion": (
            "Graduado con el menor nivel de bienestar y satisfacción del grupo. "
            "Sus competencias cognitivas (F1) y tecnológicas (F3) están en desarrollo, "
            "con puntuaciones bajas en segunda lengua, inserción laboral y herramientas "
            "modernas. Su satisfacción con la correspondencia entre formación y empleo (F2) "
            "es la más baja de los tres perfiles. Representa el grupo prioritario para "
            "intervención y seguimiento institucional."
        ),
        "fortalezas": ["lectura_academica", "trabajo_equipo", "com_escrita"],
        "debilidades": ["segunda_lengua", "insercion_laboral", "herramientas_modernas"],
        "factor_dominante": "F1 bajo — F2 bajo — F3 bajo",
        "recomendacion": (
            "Intervención prioritaria en programas de inglés, empleabilidad y "
            "herramientas digitales. Seguimiento post-grado personalizado."
        ),
    },
    1: {
        "nombre": "El Profesional Consolidado",
        "descripcion": (
            "Perfil equilibrado y sólido. Competencias cognitivas (F1) y satisfacción "
            "laboral (F2) en rango medio-alto. Mantiene brechas en segunda lengua y "
            "herramientas modernas, pero en menor grado que el Arquetipo 0. "
            "Representa al profesional que aprovechó la formación para insertarse "
            "exitosamente en el mercado laboral con una percepción positiva de su trayectoria."
        ),
        "fortalezas": ["toma_decisiones", "etica", "trabajo_equipo"],
        "debilidades": ["segunda_lengua", "herramientas_modernas", "insercion_laboral"],
        "factor_dominante": "F1 medio — F2 medio — F3 medio",
        "recomendacion": (
            "Potenciar segunda lengua y actualización tecnológica para elevar "
            "la competitividad en el mercado laboral internacional."
        ),
    },
    2: {
        "nombre": "El Líder de Alto Desempeño",
        "descripcion": (
            "El perfil más frecuente (52.7%). Competencias muy altas en las tres "
            "dimensiones factoriales: cognitivas (F1), satisfacción laboral (F2) y "
            "tecnológicas (F3). Sobresale en toma de decisiones, trabajo en equipo y ética. "
            "Incluso en segunda lengua —la brecha transversal— obtiene la mayor puntuación. "
            "Representa el modelo de egresado que la formación USTA aspira a consolidar."
        ),
        "fortalezas": ["toma_decisiones", "trabajo_equipo", "etica"],
        "debilidades": ["segunda_lengua", "insercion_laboral", "herramientas_modernas"],
        "factor_dominante": "F1 alto — F2 alto — F3 alto",
        "recomendacion": (
            "Capitalizar este perfil como referente institucional. "
            "Explorar programas de mentoría entre graduados para transferir "
            "buenas prácticas hacia los arquetipos 0 y 1."
        ),
    },
}

COLS_COMPETENCIAS = [
    "com_escrita", "com_oral", "pensamiento_critico", "metodos_cuantitativos",
    "metodos_cualitativos", "lectura_academica", "argumentacion", "segunda_lengua",
    "creatividad", "resolucion_conflictos", "liderazgo", "toma_decisiones",
    "resolucion_problemas", "investigacion", "herramientas_informaticas",
    "contextos_multiculturales", "insercion_laboral", "herramientas_modernas",
    "gestion_informacion", "trabajo_equipo", "aprendizaje_autonomo",
    "conocimientos_multidisciplinares", "etica",
]

COLS_BIENESTAR = [
    "adquirio_bienes", "mejoro_vivienda", "mejoro_salud",
    "acceso_seguridad_social", "incremento_cultural",
    "satisfecho_ocio", "red_amigos",
]

COLS_SATISFACCION = [
    "satisfaccion_vida", "satisfaccion_formacion",
    "efecto_calidad_vida", "score_bienestar",
    "correspondencia_primer_empleo", "correspondencia_empleo_actual",
]

COLS_TRAYECTORIA = [
    "nivel_cargo_actual", "salario_actual",
    "tiempo_primer_empleo", "nivel_formacion",
]

FACTORES_AFE = {
    "F1": {
        "nombre": "Competencias Cognitivas y Comunicativas",
        "variables_clave": ["argumentacion", "pensamiento_critico", "com_escrita",
                            "com_oral", "lectura_academica"],
        "varianza": 21.6,
    },
    "F2": {
        "nombre": "Satisfacción y Correspondencia Laboral",
        "variables_clave": ["correspondencia_primer_empleo", "correspondencia_empleo_actual",
                            "efecto_calidad_vida", "satisfaccion_vida"],
        "varianza": 11.2,
    },
    "F3": {
        "nombre": "Competencias Tecnológicas e Inserción Laboral",
        "variables_clave": ["herramientas_modernas", "insercion_laboral",
                            "gestion_informacion", "herramientas_informaticas"],
        "varianza": 17.7,
    },
}

# Valores válidos para cada categórica
VALS_VALIDOS_CAT = {
    "cat_genero":            {"Masculino", "Femenino", "No binario"},
    "cat_sede":              {"Bogotá", "Bucaramanga", "Villavicencio", "Tunja",
                              "Medellín", "Educación abierta y a distancia"},
    "cat_estado_civil":      {"Soltero", "Casado", "Unión libre", "Separado",
                              "Viudo", "Religioso"},
    "cat_nivel_educ_padres": {"Sin_estudios", "Basica", "Media", "Tecnico",
                              "Universitario", "Posgrado"},
    "cat_recomendaria":      {"Si", "No"},
    "cat_estudiaria_otra_vez": {"Si", "No", "No lo sabe"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _es_numero(s: str) -> bool:
    """Detecta si un string es un número (artefacto del parquet)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _limpiar_categorica(serie: pd.Series, valores_validos: set) -> pd.Series:
    """
    Limpia una columna categórica:
    - Elimina strings numéricos (coordenadas ACM almacenadas como string)
    - Elimina 'nan', 'None', ''
    - Conserva solo valores en el conjunto de válidos
    """
    def _limpiar(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s in ("nan", "None", "") or _es_numero(s):
            return np.nan
        return s if s in valores_validos else np.nan

    return serie.apply(_limpiar)


def _to_numeric_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    result = df[[c for c in cols if c in df.columns]].copy()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_base(filename: str = "base_con_arquetipos.parquet") -> pd.DataFrame:
    ruta = ARTIFACTS_PATH / filename
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró: {ruta}\nEjecuta primero: python src/train.py"
        )
    df = pd.read_parquet(ruta)
    log.info(f"Base con arquetipos cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    log.info(f"Arquetipos encontrados: {sorted(df['arquetipo'].unique())}")
    return df


def distribucion_arquetipos(df: pd.DataFrame) -> tuple:
    counts = df["arquetipo"].value_counts().sort_index()
    return counts, counts.sum()


def perfil_competencias(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_COMPETENCIAS if c in df.columns]
    X = _to_numeric_df(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def perfil_bienestar(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_BIENESTAR if c in df.columns]
    if not cols:
        return pd.DataFrame()
    X = _to_numeric_df(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3) * 100
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def perfil_satisfaccion(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_SATISFACCION if c in df.columns]
    if not cols:
        return pd.DataFrame()
    X = _to_numeric_df(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def perfil_trayectoria(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_TRAYECTORIA if c in df.columns]
    if not cols:
        return pd.DataFrame()
    X = _to_numeric_df(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def perfil_categoricas(df: pd.DataFrame) -> None:
    """
    Distribución de variables categóricas por arquetipo.
    Limpia coordenadas ACM numéricas almacenadas como string antes de calcular.
    """
    cat_cols = {
        "cat_genero":            "Género",
        "cat_sede":              "Sede",
        "cat_estado_civil":      "Estado Civil",
        "cat_nivel_educ_padres": "Nivel Educativo Padres",
        "cat_recomendaria":      "Recomendaría USTA",
        "cat_estudiaria_otra_vez": "Estudiaría otra vez",
    }

    nombres_arq = df["arquetipo"].map(
        lambda x: ARQUETIPOS.get(x, {}).get("nombre", f"Arq {x}")
    )

    print(f"\n🏷️  PERFIL CATEGÓRICO POR ARQUETIPO")
    for col, nombre in cat_cols.items():
        if col not in df.columns:
            continue

        validos = VALS_VALIDOS_CAT.get(col, set())
        serie_limpia = _limpiar_categorica(df[col], validos)
        n_validos = serie_limpia.notna().sum()

        if n_validos == 0:
            print(f"\n  {nombre}: ⚠️  Sin datos válidos")
            continue

        tabla = pd.crosstab(
            nombres_arq[serie_limpia.notna()],
            serie_limpia.dropna(),
            normalize="index"
        ).round(3) * 100

        print(f"\n  {nombre} (n={n_validos:,}):")
        print(tabla.to_string())


def analisis_factores_afe() -> None:
    sep = "─" * 55
    print(f"\n🔬 ESTRUCTURA FACTORIAL (AFE — 3 factores, Oblimin, Spearman)")
    print(sep)
    var_total = sum(f["varianza"] for f in FACTORES_AFE.values())
    print(f"  Varianza total explicada: {var_total:.1f}%")
    print(f"  KMO = 0.9530 (Excelente) | Bartlett p ≈ 0")
    print()
    for fid, info in FACTORES_AFE.items():
        print(f"  {fid} — {info['nombre']} ({info['varianza']:.1f}%)")
        print(f"       Variables clave: {', '.join(info['variables_clave'])}")


def brecha_segunda_lengua(df: pd.DataFrame) -> None:
    if "segunda_lengua" not in df.columns:
        return
    col = pd.to_numeric(df["segunda_lengua"], errors="coerce")
    print(f"\n🌐 HALLAZGO CLAVE — SEGUNDA LENGUA (brecha transversal)")
    print(f"   Media global : {col.mean():.2f} / 5.0")
    print(f"   43% calificó con 1 o 2 — presente en los 3 arquetipos")
    print(f"   Distribución:")
    for v in [1, 2, 3, 4, 5]:
        n = (col == v).sum()
        pct = n / col.notna().sum() * 100
        barra = "█" * int(pct / 2)
        print(f"   {v} — {n:4d} ({pct:.1f}%) {barra}")
    print()
    print(f"   Por arquetipo:")
    for arq in sorted(df["arquetipo"].unique()):
        sub = pd.to_numeric(df[df["arquetipo"] == arq]["segunda_lengua"], errors="coerce")
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        print(f"   {arq} — {nombre:<38}: {sub.mean():.2f}")


def movilidad_social_por_arquetipo(df: pd.DataFrame) -> None:
    if "movilidad_social" not in df.columns:
        return
    print(f"\n📈 MOVILIDAD SOCIAL POR ARQUETIPO")
    for arq in sorted(df["arquetipo"].unique()):
        sub = pd.to_numeric(
            df[df["arquetipo"] == arq]["movilidad_social"], errors="coerce"
        ).dropna()
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        ascenso    = (sub > 0).sum()
        sin_cambio = (sub == 0).sum()
        descenso   = (sub < 0).sum()
        total = len(sub)
        print(f"   {arq} — {nombre}")
        print(f"      Ascenso   : {ascenso:4d} ({ascenso/total*100:.1f}%)")
        print(f"      Sin cambio: {sin_cambio:4d} ({sin_cambio/total*100:.1f}%)")
        print(f"      Descenso  : {descenso:4d} ({descenso/total*100:.1f}%)")


def imprimir_reporte_completo(df: pd.DataFrame) -> None:
    sep = "=" * 65
    counts, total = distribucion_arquetipos(df)

    print(f"\n{sep}")
    print("  REPORTE DE ARQUETIPOS — ESTUDIO PENSER EGRESADOS USTA")
    print(f"  Metodología: AFE (Spearman·Oblimin) + ACM + Ward k=3")
    print(sep)

    print(f"\n👥 DISTRIBUCIÓN DE ARQUETIPOS (n={total:,})")
    for arq, n in counts.items():
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        pct = n / total * 100
        barra = "█" * int(pct / 2)
        print(f"   {arq} — {nombre:<38}: {n:4d} ({pct:.1f}%) {barra}")

    analisis_factores_afe()

    print(f"\n📋 DESCRIPCIÓN DE ARQUETIPOS")
    for arq in sorted(df["arquetipo"].unique()):
        info   = ARQUETIPOS.get(arq, {})
        nombre = info.get("nombre", f"Arquetipo {arq}")
        n = counts.get(arq, 0)
        print(f"\n  ── Arquetipo {arq}: {nombre} (n={n}, {n/total*100:.1f}%) ──")
        print(f"  {info.get('descripcion', '')}")
        print(f"  📊 Factores  : {info.get('factor_dominante', '')}")
        print(f"  ✅ Fortalezas: {', '.join(info.get('fortalezas', []))}")
        print(f"  ⚠️  Brechas   : {', '.join(info.get('debilidades', []))}")
        print(f"  💡 Acción    : {info.get('recomendacion', '')}")

    print(f"\n📊 MEDIAS DE COMPETENCIAS POR ARQUETIPO (escala 1–5)")
    perfil = perfil_competencias(df)
    if not perfil.empty:
        print(perfil.T.to_string())

    print(f"\n🏠 INDICADORES DE BIENESTAR POR ARQUETIPO (% respondió Sí)")
    bien = perfil_bienestar(df)
    if not bien.empty:
        print(bien.T.to_string())

    print(f"\n😊 SATISFACCIÓN Y CALIDAD DE VIDA")
    sat = perfil_satisfaccion(df)
    if not sat.empty:
        print(sat.T.to_string())

    print(f"\n💼 TRAYECTORIA LABORAL")
    tray = perfil_trayectoria(df)
    if not tray.empty:
        print(tray.T.to_string())

    perfil_categoricas(df)

    brecha_segunda_lengua(df)
    movilidad_social_por_arquetipo(df)

    print(f"\n{sep}")
    print("  RECOMENDACIONES INSTITUCIONALES")
    print(sep)
    print("""
  1. URGENTE — Segunda lengua: brecha transversal en los 3 arquetipos.
     Media 2.76/5 y 43% calificó con 1 o 2. Reforzar programas de
     inglés de forma obligatoria y transversal a todos los programas.

  2. ALTA — Inserción laboral y herramientas modernas: segundas brechas.
     Fortalecer empleabilidad y actualización tecnológica, especialmente
     para el Arquetipo 0 (El Subjetivamente Satisfecho).

  3. ESTRATÉGICO — El 52.7% son Líderes de Alto Desempeño (F1+F2+F3 altos).
     Capitalizar este perfil con programas de mentoría entre graduados
     para elevar a los arquetipos 0 y 1.

  4. SEGUIMIENTO — El Arquetipo 0 (23.5%) es el grupo de mayor riesgo.
     Implementar seguimiento post-grado diferenciado por arquetipo
     en lugar de estrategias institucionales genéricas.
    """)
    print(f"{sep}\n")


def guardar_reportes(df: pd.DataFrame) -> None:
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    perfil = perfil_competencias(df)
    if not perfil.empty:
        perfil.T.to_csv(ARTIFACTS_PATH / "perfil_competencias_arquetipos.csv")
        log.info("Perfil competencias guardado.")

    bien = perfil_bienestar(df)
    if not bien.empty:
        bien.T.to_csv(ARTIFACTS_PATH / "perfil_bienestar_arquetipos.csv")
        log.info("Perfil bienestar guardado.")

    sat = perfil_satisfaccion(df)
    if not sat.empty:
        sat.T.to_csv(ARTIFACTS_PATH / "perfil_satisfaccion_arquetipos.csv")
        log.info("Perfil satisfacción guardado.")

    df_out = df.copy()
    df_out["nombre_arquetipo"] = df_out["arquetipo"].map(
        lambda x: ARQUETIPOS.get(x, {}).get("nombre", f"Arquetipo {x}")
    )
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
    df_out.to_parquet(ARTIFACTS_PATH / "base_con_arquetipos.parquet", index=False)
    log.info("Base final actualizada.")


def run() -> None:
    log.info("Iniciando evaluate.py...")
    df = cargar_base()
    imprimir_reporte_completo(df)
    guardar_reportes(df)
    log.info("Evaluación completada.")


if __name__ == "__main__":
    run()