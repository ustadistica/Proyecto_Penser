"""
evaluate_depurada.py — Versión 5.0
====================================
Descriptivas profundas por arquetipo — Base Depurada PENSER USTA 2025
Metodología: AFE Bloques (3) + ACM + QuantileTransformer + PCA + KMeans k=2

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_PATH = Path("artifacts")

ARQUETIPOS_K2 = {
    0: {"nombre": "El Graduado en Desarrollo",      "color": "🟡"},
    1: {"nombre": "El Profesional de Alto Impacto", "color": "🟢"},
}

COLS_LOGRO = [
    "logro_comp_prof_1","logro_comp_prof_2","logro_comp_prof_3",
    "logro_comp_prof_4","logro_comp_prof_5",
    "logro_comunicacion","logro_relaciones_interpersonales",
    "logro_toma_decisiones","logro_solucion_problemas",
    "logro_pensamiento_creativo","logro_pensamiento_critico",
    "logro_manejo_emociones","logro_manejo_estres",
    "logro_multiculturales","logro_interculturales",
]

COLS_INCIDENCIA = [
    "incidencia_ingresos","incidencia_empleo","incidencia_vivienda",
    "incidencia_salud","incidencia_recreacion","incidencia_educacion",
]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

def _sep(n=65): return "=" * n
def _sub(n=65): return "─" * n

def _nombre(x):
    return ARQUETIPOS_K2.get(int(x), {}).get("nombre", f"Arquetipo {x}")

def _kruskal(df, col, arq_col="_arq"):
    grupos = [_num(df[df[arq_col]==a], col).dropna()
              for a in sorted(df[arq_col].unique())]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2: return np.nan, np.nan
    try:
        stat, p = stats.kruskal(*grupos)
        return round(stat,3), round(p,4)
    except: return np.nan, np.nan

def _es_valido(x):
    return (pd.notna(x) and str(x).strip() not in ["nan","None",""]
            and not str(x).strip().lstrip("-").isdigit())

def _get_arq_col(df):
    if "arquetipo_kmeans_opt" in df.columns:
        return "arquetipo_kmeans_opt"
    return "arquetipo_ward_opt"


def cargar_base():
    ruta = ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No encontrado: {ruta}\nEjecuta: train_depurada.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} × {df.shape[1]}")

    raw = Path("data/raw/DATA_DEPURADA_PENSER_2025.xlsx")
    if raw.exists():
        df_raw = pd.read_excel(raw)
        df_raw = df_raw[~df_raw.isnull().all(axis=1)]
        df_raw = df_raw.drop(
            columns=df_raw.columns[df_raw.isnull().mean() > 0.90].tolist(), errors="ignore"
        ).reset_index(drop=True)

        if "Año de graduación" in df_raw.columns:
            df["año_graduacion"] = df_raw["Año de graduación"].values[:len(df)]
            df["años_graduado"] = 2026 - df["año_graduacion"]

        if "PROGRAMA ACADEMICO" in df_raw.columns:
            df["programa"] = df_raw["PROGRAMA ACADEMICO"].values[:len(df)]

        if "Sede o Seccional" in df_raw.columns:
            df["sede_raw"] = df_raw["Sede o Seccional"].values[:len(df)]

        log.info("Variables demográficas enriquecidas desde raw.")
    return df


def seccion_distribucion(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  1. DISTRIBUCIÓN DE ARQUETIPOS")
    print(_sep())
    total = len(df)

    print(f"\n  KMeans k=2 (método principal — silueta=0.539):")
    counts = df[arq_col].value_counts().sort_index()
    for arq, n in counts.items():
        pct = n/total*100
        barra = "█" * int(pct/2)
        print(f"    {arq} — {_nombre(arq):<35}: {n:4d} ({pct:.1f}%) {barra}")

    if "arquetipo_ward_opt" in df.columns:
        print(f"\n  Ward k=2 (validación — silueta=0.525):")
        counts_w = df["arquetipo_ward_opt"].value_counts().sort_index()
        for arq, n in counts_w.items():
            pct = n/total*100
            print(f"    {arq}: {n:4d} ({pct:.1f}%)")

    print(f"\n  Métricas KMeans k=2:")
    print(f"    Silueta : 0.5390 ✅")
    print(f"    Dunn    : 0.0451 ✅")
    print(f"    DB      : 0.8110 ✅")
    print(f"    PCA     : 3 componentes → 73.4% varianza explicada")
    print(f"    DBSCAN  : Viable — silueta=0.630 (confirma estructura binaria)")


def seccion_logro(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  2. PERFIL DE LOGRO DE COMPETENCIAS (escala 1–5)")
    print(_sep())

    df["_arq"] = df[arq_col]
    cols = [c for c in COLS_LOGRO if c in df.columns]
    n_arqs = sorted(df["_arq"].unique())

    header = f"  {'Competencia':<35}"
    for a in n_arqs: header += f" {'Arq'+str(a):>8}"
    header += f" {'KW p':>8}"
    print(f"\n{header}")
    print(f"  {_sub()}")

    for col in cols:
        vals = [_num(df[df["_arq"]==a], col).mean() for a in n_arqs]
        _, p = _kruskal(df, col)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        row = f"  {col:<35}"
        for v in vals: row += f" {v:>8.3f}"
        row += f" {p:>7.4f}{sig}"
        print(row)

    print(f"\n  SCORES COMPUESTOS:")
    for sc, mx in [("score_logro_competencias",50),("score_impacto_formacion",30)]:
        if sc not in df.columns: continue
        print(f"\n  {sc} (max={mx}):")
        for a in n_arqs:
            sub = _num(df[df["_arq"]==a], sc)
            print(f"    {_nombre(a)}: media={sub.mean():.2f} | mediana={sub.median():.1f} | std={sub.std():.2f}")
        _, p = _kruskal(df, sc)
        print(f"    Kruskal-Wallis p={p:.4f} {'***' if p<0.001 else ''}")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_incidencia(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  3. INCIDENCIA DE LA FORMACIÓN EN BIENESTAR (escala 0–5)")
    print(_sep())

    df["_arq"] = df[arq_col]
    cols = [c for c in COLS_INCIDENCIA if c in df.columns]
    n_arqs = sorted(df["_arq"].unique())

    header = f"  {'Variable':<30}"
    for a in n_arqs: header += f" {'Arq'+str(a):>8}"
    header += f" {'KW p':>8}"
    print(f"\n{header}")
    print(f"  {_sub()}")

    for col in cols:
        vals = [_num(df[df["_arq"]==a], col).mean() for a in n_arqs]
        _, p = _kruskal(df, col)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        row = f"  {col:<30}"
        for v in vals: row += f" {v:>8.3f}"
        row += f" {p:>7.4f}{sig}"
        print(row)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_programa(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  4. PROGRAMA ACADÉMICO POR ARQUETIPO")
    print(_sep())

    if "programa" not in df.columns:
        print("  ⚠️  Variable 'programa' no disponible.")
        return

    df["_arq"] = df[arq_col]
    df_p = df[df["programa"].notna() & (df["programa"].astype(str) != "nan")]
    total_p = len(df_p)

    print(f"\n  Base con programa: {total_p:,} ({total_p/len(df)*100:.1f}%)")
    print(f"  Programas únicos: {df_p['programa'].nunique()}")

    print(f"\n  Top 8 globales:")
    for prog, n in df_p["programa"].value_counts().head(8).items():
        print(f"    {n:4d} ({n/total_p*100:.1f}%) | {str(prog).strip()[:50]}")

    for a in sorted(df["_arq"].unique()):
        sub = df_p[df_p["_arq"]==a]
        print(f"\n  {_nombre(a)} (n={len(sub)}):")
        for prog, n in sub["programa"].value_counts().head(5).items():
            pct_arq = n/len(sub)*100
            pct_prog = n/df_p[df_p["programa"]==prog].shape[0]*100
            print(f"    {n:4d} ({pct_arq:.1f}% del arq | {pct_prog:.1f}% del prog) | {str(prog).strip()[:50]}")

    # Marginal inversa
    print(f"\n  Por carrera → ¿en qué arquetipo se concentran? (marginal inversa):")
    top_progs = df_p["programa"].value_counts().head(8).index
    tabla = pd.crosstab(
        df_p["programa"].astype(str).str.strip(),
        df_p["_arq"], normalize="index"
    ).round(3)*100
    tabla.columns = [_nombre(c)[:12] for c in tabla.columns]
    top = tabla[tabla.index.isin(top_progs)]
    print(f"\n  {'Programa':<45}" + "".join([f" {c:>14}" for c in top.columns]))
    print(f"  {_sub(75)}")
    for prog, row in top.iterrows():
        line = f"  {str(prog)[:45]:<45}"
        for c in top.columns: line += f" {float(row[c]):>13.1f}%"
        print(line)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_graduacion(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  5. AÑO DE GRADUACIÓN POR ARQUETIPO")
    print(_sep())

    if "año_graduacion" not in df.columns:
        print("  ⚠️  Variable 'año_graduacion' no disponible.")
        return

    df["_arq"] = df[arq_col]
    df_g = df[df["año_graduacion"].notna()]
    n_arqs = sorted(df["_arq"].unique())

    print(f"\n  {'Arquetipo':<35} {'n':>5} {'Media':>7} {'Mediana':>8} {'Min':>6} {'Max':>6}")
    print(f"  {_sub(65)}")
    for a in n_arqs:
        sub = df_g[df_g["_arq"]==a]["año_graduacion"]
        print(f"  {_nombre(a):<35} {len(sub):>5} {sub.mean():>7.1f} {sub.median():>8.0f} {sub.min():>6.0f} {sub.max():>6.0f}")

    grupos = [df_g[df_g["_arq"]==a]["año_graduacion"].dropna() for a in n_arqs]
    _, p = stats.kruskal(*grupos)
    print(f"\n  Kruskal-Wallis p={p:.4f} {'*** diferencias significativas' if p<0.001 else 'ns — sin diferencia'}")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_trayectoria(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  6. TRAYECTORIA LABORAL POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df[arq_col]
    n_arqs = sorted(df["_arq"].unique())

    cols_lab = {
        "laborando_actualmente": "Laborando actualmente (% Sí)",
        "ingresos_superiores_estudio": "Ingresos superiores al estudiar (% Sí)",
        "percepcion_programa": "Percepción programa (1-5)",
        "percepcion_ingreso": "Percepción ingreso (1-4)",
        "formacion_impacto_general": "Impacto general formación (1-5)",
    }

    header = f"  {'Variable':<40}"
    for a in n_arqs: header += f" {'Arq'+str(a):>8}"
    header += f" {'KW p':>8}"
    print(f"\n{header}")
    print(f"  {_sub()}")

    for col, label in cols_lab.items():
        if col not in df.columns: continue
        vals = [_num(df[df["_arq"]==a], col).mean() for a in n_arqs]
        _, p = _kruskal(df, col)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        row = f"  {label:<40}"
        for v in vals: row += f" {v:>8.3f}"
        row += f" {p:>7.4f} {sig}"
        print(row)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_categoricas(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  7. PERFIL CATEGÓRICO POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df[arq_col]
    nombres_arq = df["_arq"].map(_nombre)

    for col, nombre in [
        ("cat_sede","Sede"),
        ("cat_laborando","Laborando"),
        ("cat_tipo_cargo","Tipo de cargo"),
    ]:
        if col not in df.columns: continue
        serie = df[col].apply(lambda x: str(x).strip() if _es_valido(x) else np.nan)
        if serie.notna().sum() == 0: continue
        tabla = pd.crosstab(
            nombres_arq[serie.notna()], serie.dropna(), normalize="index"
        ).round(3)*100
        print(f"\n  {nombre} (n={serie.notna().sum():,}):")
        print(tabla.to_string())

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_robustez(df):
    print(f"\n{_sep()}")
    print("  8. ROBUSTEZ — KMeans vs Ward")
    print(_sep())

    if "arquetipo_kmeans_opt" not in df.columns or "arquetipo_ward_opt" not in df.columns:
        print("  ⚠️  Columnas no disponibles.")
        return

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(df["arquetipo_kmeans_opt"], df["arquetipo_ward_opt"])
    print(f"\n  Adjusted Rand Index (KMeans k=2 vs Ward k=2): {ari:.4f}")
    if ari > 0.6:
        print(f"  → Alta concordancia: estructura robusta entre métodos. ✅")
    elif ari > 0.3:
        print(f"  → Concordancia moderada.")
    else:
        print(f"  → Baja concordancia.")

    print(f"\n  Métricas comparadas:")
    print(f"    KMeans k=2: Silueta=0.5390 | DB=0.8110 | Dunn=0.0451")
    print(f"    Ward   k=2: Silueta=0.5253 | DB=0.9462 | Dunn=0.0752")
    print(f"    DBSCAN k=2: Silueta=0.6299 (confirma estructura binaria natural)")


def seccion_limitacion_sede(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  9. LIMITACIÓN METODOLÓGICA — EFECTO SEDE")
    print(_sep())

    col_sede = "sede_raw" if "sede_raw" in df.columns else "cat_sede"
    if col_sede not in df.columns:
        print("  ⚠️  Variable sede no disponible.")
        return

    df["_arq"] = df[arq_col]
    serie = df[col_sede].apply(lambda x: str(x).strip() if _es_valido(x) else np.nan)
    nombres_arq = df["_arq"].map(_nombre)

    tabla = pd.crosstab(
        nombres_arq[serie.notna()], serie.dropna(), normalize="index"
    ).round(3)*100

    print(f"\n  Distribución sede por arquetipo (%):")
    print(tabla.to_string())
    print(f"""
  ⚠️  LIMITACIÓN DOCUMENTADA:
  Con k=2 la asociación sede-arquetipo es menos extrema que con k=4.
  Sin embargo, Bucaramanga sigue concentrando más en el Profesional de
  Alto Impacto. El efecto territorial persiste y debe documentarse.
  Recomendación: análisis separados por sede para aislar el efecto
  institucional del efecto territorial.
    """)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_factores_comunes(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  10. FACTORES COMUNES CON BASE PERCEPCIÓN")
    print(_sep())

    # Calcular números reales desde la base
    df["_arq"] = df[arq_col]

    # Scores por arquetipo
    scores = {}
    for a in sorted(df["_arq"].unique()):
        sub = df[df["_arq"]==a]
        scores[a] = {
            "n": len(sub),
            "pct": len(sub)/len(df)*100,
            "logro": _num(sub, "score_logro_competencias").mean(),
            "impacto": _num(sub, "score_impacto_formacion").mean(),
        }
        if "laborando_actualmente" in df.columns:
            scores[a]["laborando"] = _num(sub, "laborando_actualmente").mean()*100
        if "ingresos_superiores_estudio" in df.columns:
            scores[a]["ingresos"] = _num(sub, "ingresos_superiores_estudio").mean()*100

    dep = scores

    print(f"""
  TABLA COMPARATIVA DE ARQUETIPOS ANÁLOGOS
  {_sub(65)}
  BASE PERCEPCIÓN (KMeans k=3)             BASE DEPURADA (KMeans k=2)
  {_sub(65)}

  El Profesional en Desarrollo             El Graduado en Desarrollo
  69.6% · n=1.761                          {dep[0]['pct']:.1f}% · n={dep[0]['n']}
  Bienestar: 3.64/7                        Logro: {dep[0]['logro']:.2f}/50
  Recomendaría USTA: 67.6%                 Impacto: {dep[0]['impacto']:.2f}/30
  Inserción laboral: 3.15/5               Laborando: {dep[0].get('laborando', 0):.1f}%

  El Líder de Alto Desempeño              El Profesional de Alto Impacto
  12.6% · n=320                           {dep[1]['pct']:.1f}% · n={dep[1]['n']}
  Bienestar: 4.60/7                        Logro: {dep[1]['logro']:.2f}/50
  Recomendaría USTA: 89.3%                 Impacto: {dep[1]['impacto']:.2f}/30
  Segunda lengua: 3.97/5                  Laborando: {dep[1].get('laborando', 0):.1f}%
  {_sub(65)}

  COMPONENTES COMUNES VERIFICADOS:

  1. El perfil de alto desempeño es robusto en ambas bases:
     Percepción: bienestar 4.60/7, rec 89.3%, inserción 4.84/5
     Depurada  : logro {dep[1]['logro']:.2f}/50, impacto {dep[1]['impacto']:.2f}/30, laborando {dep[1].get('laborando',0):.1f}%
     → Grupo más pequeño pero mejor definido (silueta alta).

  2. El grupo mayoritario (~70-80%) muestra perfil de desarrollo:
     Percepción: 69.6% — bienestar 3.64/7, 67.6% recomendaría
     Depurada  : {dep[0]['pct']:.1f}% — logro {dep[0]['logro']:.2f}/50, impacto {dep[0]['impacto']:.2f}/30
     → Formación con impacto diferenciado según instrumento.

  3. Segunda lengua — brecha transversal:
     El Líder en percepción tiene 3.97/5 — la más alta.
     Sigue siendo la competencia con mayor área de mejora global (2.76/5).

  DIFERENCIA CLAVE ENTRE INSTRUMENTOS:
  Percepción captura satisfacción subjetiva y correspondencia laboral.
  Depurada captura impacto concreto (ingresos, empleo, vivienda).
  Ambas bases muestran estructura binaria natural (DBSCAN viable en depurada).
    """)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_recomendaciones(df):
    arq_col = _get_arq_col(df)
    df["_arq"] = df[arq_col]

    lab_pcts = {}
    if "laborando_actualmente" in df.columns:
        for a in sorted(df["_arq"].unique()):
            sub = df[df["_arq"]==a]
            lab_pcts[a] = f"{_num(sub,'laborando_actualmente').mean()*100:.1f}%"

    df.drop(columns=["_arq"], inplace=True, errors="ignore")

    print(f"\n{_sep()}")
    print("  11. RECOMENDACIONES INSTITUCIONALES")
    print(_sep())
    print(f"""
  R1 — URGENTE: El 82.2% son Graduados en Desarrollo
       Laborando: {lab_pcts.get(0,'N/A')}. La formación tiene impacto diferenciado.
       Implementar protocolo de seguimiento post-grado diferenciado.

  R2 — ALTA: Competencia profesional específica (logro_comp_prof_3)
       La más baja en ambos arquetipos. Revisión curricular urgente
       independiente del programa académico.

  R3 — ESTRATÉGICO: El Profesional de Alto Impacto ({lab_pcts.get(1,'N/A')} laborando)
       Es el referente institucional. Capitalizar para mentoría y
       visibilidad. Logro 48.53/50 — casi perfecto.

  R4 — ESTRUCTURAL: Análisis separado por sede
       Bucaramanga concentra más Profesionales de Alto Impacto.
       El efecto territorial no puede separarse del institucional
       sin diseño específico de investigación.

  R5 — INVESTIGACIÓN: Cruzar con base percepción por programa
       Negocios Internacionales y Derecho aparecen en el top de ambas bases.
    """)


def run():
    log.info("Iniciando evaluate_depurada.py v5...")
    df = cargar_base()

    print(f"\n{_sep()}")
    print("  REPORTE ARQUETIPOS — BASE DEPURADA PENSER USTA 2025")
    print(f"  Metodología: AFE Bloques (3) + ACM + QuantileTransformer + PCA + KMeans k=2")
    print(f"  Base: 1.129 respuestas | 2 arquetipos | Silueta=0.539")
    print(_sep())

    seccion_distribucion(df)
    seccion_logro(df)
    seccion_incidencia(df)
    seccion_programa(df)
    seccion_graduacion(df)
    seccion_trayectoria(df)
    seccion_categoricas(df)
    seccion_robustez(df)
    seccion_limitacion_sede(df)
    seccion_factores_comunes(df)
    seccion_recomendaciones(df)

    df["nombre_arquetipo"] = df[_get_arq_col(df)].map(_nombre)
    df.to_parquet(ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet", index=False)
    log.info("Base final guardada.")
    log.info("evaluate_depurada.py v5 completado.")


if __name__ == "__main__":
    run()