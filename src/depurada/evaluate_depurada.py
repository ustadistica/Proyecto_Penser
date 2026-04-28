"""
evaluate_depurada.py
====================
Descripción, interpretación y reporte de arquetipos — Base Depurada PENSER USTA 2025.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Arquetipos identificados (3) mediante AFE + ACM + Ward:
---------------------------------------------------------
0 — El Graduado en Desarrollo     : 587 (52.0%) — logro bajo, incidencia baja
1 — El Profesional Impactado      : 242 (21.4%) — logro medio, incidencia media
2 — El Líder con Alta Incidencia  : 300 (26.6%) — logro alto, incidencia alta

Factores AFE:
--------------
F1 — Competencias Transversales (34.3%): emociones, estrés, pensamiento, relaciones
F2 — Incidencia en Bienestar (19.5%): salud, vivienda, ingresos, empleo
F3 — Competencias Profesionales del Programa (14.5%): logro_comp_prof 1-5
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

ARTIFACTS_PATH = Path("artifacts")

# ---------------------------------------------------------------------------
# Definición de arquetipos
# ---------------------------------------------------------------------------
ARQUETIPOS = {
    0: {
        "nombre": "El Graduado en Desarrollo",
        "descripcion": (
            "Graduado con el menor logro en competencias y la menor incidencia "
            "percibida de la formación en su bienestar. Sus competencias transversales "
            "(F1) y profesionales (F3) están en rango bajo-medio, y siente que la "
            "formación tuvo poca incidencia en ingresos, vivienda o salud (F2 bajo, "
            "score incidencia ~6.9/30). Representa el grupo con mayor potencial de "
            "mejora y mayor necesidad de seguimiento institucional."
        ),
        "fortalezas": ["logro_pensamiento_critico", "logro_comunicacion"],
        "debilidades": ["logro_manejo_estres", "incidencia_vivienda", "incidencia_salud"],
        "factor_dominante": "F1 bajo — F2 bajo — F3 bajo",
        "recomendacion": (
            "Fortalecer programas de bienestar estudiantil, orientación vocacional "
            "y seguimiento post-grado. Revisar pertinencia curricular con el mercado laboral."
        ),
    },
    1: {
        "nombre": "El Profesional Impactado",
        "descripcion": (
            "Perfil intermedio con competencias medias (logro ~3.5/5) y una incidencia "
            "moderada de la formación en su bienestar (score incidencia ~14.7/30). "
            "Reconoce que la formación tuvo efecto en su inserción laboral e ingresos, "
            "aunque percibe que su desarrollo en competencias transversales (F1) y "
            "profesionales (F3) fue parcial. Perfil mayoritario con amplio margen de mejora."
        ),
        "fortalezas": ["logro_pensamiento_critico", "logro_toma_decisiones", "logro_comunicacion"],
        "debilidades": ["logro_comp_prof_3", "incidencia_vivienda", "incidencia_recreacion"],
        "factor_dominante": "F1 medio — F2 medio — F3 medio",
        "recomendacion": (
            "Potenciar la conexión entre competencias del programa y el mercado laboral. "
            "Fortalecer prácticas y pasantías para aumentar la incidencia real."
        ),
    },
    2: {
        "nombre": "El Líder con Alta Incidencia",
        "descripcion": (
            "Graduado con el mayor logro en competencias y la mayor incidencia percibida "
            "de la formación en su calidad de vida (score incidencia ~20.8/30). Sus "
            "competencias transversales (F1) y profesionales (F3) son altas, y reconoce "
            "que la USTA tuvo impacto concreto en su acceso a empleo, ingresos y educación. "
            "Representa el perfil de mayor éxito del modelo educativo institucional."
        ),
        "fortalezas": ["logro_pensamiento_critico", "logro_comunicacion", "logro_toma_decisiones"],
        "debilidades": ["logro_comp_prof_3", "incidencia_recreacion"],
        "factor_dominante": "F1 alto — F2 alto — F3 alto",
        "recomendacion": (
            "Capitalizar este perfil como referente. Diseñar programas de mentoría "
            "entre graduados y vincularlos a estrategias de visibilidad institucional."
        ),
    },
}

FACTORES_AFE = {
    "F1": {
        "nombre": "Competencias Transversales",
        "variables_clave": ["logro_manejo_emociones", "logro_manejo_estres",
                            "logro_pensamiento_critico", "logro_solucion_problemas"],
        "varianza": 34.3,
    },
    "F2": {
        "nombre": "Incidencia en Bienestar",
        "variables_clave": ["incidencia_salud", "incidencia_vivienda",
                            "incidencia_ingresos", "incidencia_empleo"],
        "varianza": 19.5,
    },
    "F3": {
        "nombre": "Competencias Profesionales del Programa",
        "variables_clave": ["logro_comp_prof_4", "logro_comp_prof_2",
                            "logro_comp_prof_3", "logro_comp_prof_1"],
        "varianza": 14.5,
    },
}

COLS_LOGRO = [
    "logro_comp_prof_1", "logro_comp_prof_2", "logro_comp_prof_3",
    "logro_comp_prof_4", "logro_comp_prof_5",
    "logro_comunicacion", "logro_relaciones_interpersonales",
    "logro_toma_decisiones", "logro_solucion_problemas",
    "logro_pensamiento_creativo", "logro_pensamiento_critico",
    "logro_manejo_emociones", "logro_manejo_estres",
    "logro_multiculturales", "logro_interculturales",
]

COLS_INCIDENCIA = [
    "incidencia_ingresos", "incidencia_empleo", "incidencia_vivienda",
    "incidencia_salud", "incidencia_recreacion", "incidencia_educacion",
]

COLS_BINARIAS = [
    "laborando_actualmente", "recibio_distinciones", "pertenece_gremio",
    "lidera_proyectos_comunitarios", "lidera_investigacion",
    "ingresos_superiores_estudio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_num(df, cols):
    result = df[[c for c in cols if c in df.columns]].copy()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _es_numero(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_base():
    ruta = ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró: {ruta}\n"
            f"Ejecuta primero: python src/depurada/train_depurada.py"
        )
    df = pd.read_parquet(ruta)
    log.info(f"Base con arquetipos cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    log.info(f"Arquetipos: {sorted(df['arquetipo'].unique())}")
    return df


def distribucion(df):
    counts = df["arquetipo"].value_counts().sort_index()
    return counts, counts.sum()


def perfil_logro(df):
    cols = [c for c in COLS_LOGRO if c in df.columns]
    X = _to_num(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x,{}).get('nombre',f'Arq {x}')}"
    )
    return perfil


def perfil_incidencia(df):
    cols = [c for c in COLS_INCIDENCIA if c in df.columns]
    if not cols:
        return pd.DataFrame()
    X = _to_num(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x,{}).get('nombre',f'Arq {x}')}"
    )
    return perfil


def perfil_binarias(df):
    cols = [c for c in COLS_BINARIAS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    X = _to_num(df, cols)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols].mean().round(3) * 100
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x,{}).get('nombre',f'Arq {x}')}"
    )
    return perfil


def perfil_trayectoria(df):
    cols = ["tipo_cargo", "tipo_vinculacion", "percepcion_programa",
            "percepcion_ingreso", "nivel_estudios", "formacion_impacto_general"]
    cols_ok = [c for c in cols if c in df.columns]
    if not cols_ok:
        return pd.DataFrame()
    X = _to_num(df, cols_ok)
    X["arquetipo"] = df["arquetipo"].values
    perfil = X.groupby("arquetipo")[cols_ok].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x,{}).get('nombre',f'Arq {x}')}"
    )
    return perfil


def perfil_categoricas(df):
    cat_cols = {
        "cat_sede":       "Sede",
        "cat_laborando":  "Laborando actualmente",
        "cat_tipo_cargo": "Tipo de cargo",
    }
    nombres_arq = df["arquetipo"].map(
        lambda x: ARQUETIPOS.get(x, {}).get("nombre", f"Arq {x}")
    )
    print(f"\n🏷️  PERFIL CATEGÓRICO POR ARQUETIPO")
    for col, nombre in cat_cols.items():
        if col not in df.columns:
            continue
        serie = df[col].apply(
            lambda x: np.nan if pd.isna(x) or str(x).strip() in ("nan","None","")
            or _es_numero(str(x).strip()) else str(x).strip()
        )
        if serie.notna().sum() == 0:
            continue
        tabla = pd.crosstab(
            nombres_arq[serie.notna()],
            serie.dropna(),
            normalize="index"
        ).round(3) * 100
        print(f"\n  {nombre} (n={serie.notna().sum():,}):")
        print(tabla.to_string())


def analisis_factores():
    sep = "─" * 55
    print(f"\n🔬 ESTRUCTURA FACTORIAL (AFE — 3 factores, Oblimin, Spearman)")
    print(sep)
    var_total = sum(f["varianza"] for f in FACTORES_AFE.values())
    print(f"  Varianza total explicada : {var_total:.1f}%")
    print(f"  KMO = 0.9344 (Excelente) | Bartlett p ≈ 0")
    print()
    for fid, info in FACTORES_AFE.items():
        print(f"  {fid} — {info['nombre']} ({info['varianza']:.1f}%)")
        print(f"       Variables clave: {', '.join(info['variables_clave'])}")


def scores_compuestos_por_arquetipo(df):
    print(f"\n📊 SCORES COMPUESTOS POR ARQUETIPO")
    scores = ["score_logro_competencias", "score_impacto_formacion", "score_actividad_profesional"]
    maximos = {"score_logro_competencias": 50, "score_impacto_formacion": 30,
               "score_actividad_profesional": 4}
    for arq in sorted(df["arquetipo"].unique()):
        sub = df[df["arquetipo"] == arq]
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        print(f"\n  Arquetipo {arq} — {nombre} (n={len(sub)})")
        for sc in scores:
            if sc in df.columns:
                val = pd.to_numeric(sub[sc], errors="coerce").mean()
                mx = maximos.get(sc, "?")
                print(f"    {sc:35}: {val:.2f} / {mx}")


def imprimir_reporte(df):
    sep = "=" * 65
    counts, total = distribucion(df)

    print(f"\n{sep}")
    print("  REPORTE ARQUETIPOS — BASE DEPURADA PENSER USTA 2025")
    print(f"  Metodología: AFE (Spearman·Oblimin) + ACM + Ward k=3")
    print(sep)

    print(f"\n👥 DISTRIBUCIÓN (n={total:,})")
    for arq, n in counts.items():
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        pct = n / total * 100
        barra = "█" * int(pct / 2)
        print(f"   {arq} — {nombre:<38}: {n:4d} ({pct:.1f}%) {barra}")

    analisis_factores()

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

    print(f"\n📈 LOGRO DE COMPETENCIAS POR ARQUETIPO (escala 1–5)")
    pl = perfil_logro(df)
    if not pl.empty:
        print(pl.T.to_string())

    print(f"\n🎯 INCIDENCIA DE LA FORMACIÓN EN BIENESTAR (escala 0–5)")
    pi = perfil_incidencia(df)
    if not pi.empty:
        print(pi.T.to_string())

    print(f"\n✅ ACTIVIDAD PROFESIONAL POST-GRADO (% Sí)")
    pb = perfil_binarias(df)
    if not pb.empty:
        print(pb.T.to_string())

    print(f"\n💼 TRAYECTORIA LABORAL")
    pt = perfil_trayectoria(df)
    if not pt.empty:
        print(pt.T.to_string())

    perfil_categoricas(df)
    scores_compuestos_por_arquetipo(df)

    print(f"\n{sep}")
    print("  RECOMENDACIONES INSTITUCIONALES")
    print(sep)
    print("""
  1. URGENTE — El 52% son Graduados en Desarrollo con baja incidencia.
     Revisar pertinencia del currículo con el mercado laboral real.
     Implementar seguimiento post-grado diferenciado por arquetipo.

  2. ALTA — Competencia profesional específica (logro_comp_prof_3)
     es la más baja en los 3 arquetipos. Requiere revisión curricular
     del componente profesional específico del programa.

  3. ESTRATÉGICO — El 26.6% son Líderes con Alta Incidencia.
     Capitalizar su testimonio y trayectoria para visibilizar
     el impacto real de la formación USTA.

  4. ESTRUCTURAL — La incidencia en recreación y vivienda es la
     más baja en todos los arquetipos. Señala límites del impacto
     de la formación universitaria en bienestar material inmediato.
    """)
    print(f"{sep}\n")


def guardar_reportes(df):
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    pl = perfil_logro(df)
    if not pl.empty:
        pl.T.to_csv(ARTIFACTS_PATH / "perfil_logro_arquetipos_depurada.csv")
        log.info("Perfil logro guardado.")

    pi = perfil_incidencia(df)
    if not pi.empty:
        pi.T.to_csv(ARTIFACTS_PATH / "perfil_incidencia_arquetipos_depurada.csv")
        log.info("Perfil incidencia guardado.")

    df_out = df.copy()
    df_out["nombre_arquetipo"] = df_out["arquetipo"].map(
        lambda x: ARQUETIPOS.get(x, {}).get("nombre", f"Arquetipo {x}")
    )
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
    df_out.to_parquet(ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet", index=False)
    log.info("Base final actualizada.")


def run():
    log.info("Iniciando evaluate_depurada.py...")
    df = cargar_base()
    imprimir_reporte(df)
    guardar_reportes(df)
    log.info("Evaluación completada.")


if __name__ == "__main__":
    run()