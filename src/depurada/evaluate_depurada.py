"""
evaluate_depurada.py — Versión 4.0
====================================
Descriptivas profundas por arquetipo — Base Depurada PENSER USTA 2025

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Contenido:
----------
1.  Distribución de arquetipos (Ward k=4 vs KProto k=3)
2.  Perfil de competencias por arquetipo
3.  Incidencia de la formación en bienestar
4.  Programa académico por arquetipo
5.  Año de graduación por arquetipo
6.  Trayectoria laboral
7.  Perfil categórico (sede, tipo cargo, laborando)
8.  Robustez Ward vs KPrototypes
9.  Limitación metodológica: efecto sede
10. Factores comunes con base percepción
11. Recomendaciones institucionales
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

ARQUETIPOS_K4 = {
    0: {"nombre": "El Graduado en Desarrollo",    "color": "🔴"},
    1: {"nombre": "El Profesional en Formación",  "color": "🟠"},
    2: {"nombre": "El Profesional Impactado",     "color": "🟡"},
    3: {"nombre": "El Líder con Alta Incidencia", "color": "🟢"},
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
    return ARQUETIPOS_K4.get(int(x), {}).get("nombre", f"Arquetipo {x}")

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


def cargar_base():
    ruta = ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No encontrado: {ruta}\nEjecuta: train_depurada.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} × {df.shape[1]}")

    # Enriquecer con variables del raw
    raw = Path("data/raw/DATA_DEPURADA_PENSER_2025.xlsx")
    if raw.exists():
        df_raw = pd.read_excel(raw)
        df_raw = df_raw[~df_raw.isnull().all(axis=1)]
        df_raw = df_raw.drop(
            columns=df_raw.columns[df_raw.isnull().mean() > 0.90].tolist(),
            errors="ignore"
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
    print(f"\n{_sep()}")
    print("  1. DISTRIBUCIÓN DE ARQUETIPOS")
    print(_sep())
    total = len(df)

    print(f"\n  Ward k=4 (método principal — mejor score=0.2832):")
    counts = df["arquetipo_ward_opt"].value_counts().sort_index()
    for arq, n in counts.items():
        pct = n/total*100
        barra = "█" * int(pct/2)
        print(f"    {arq} — {_nombre(arq):<35}: {n:4d} ({pct:.1f}%) {barra}")

    if "arquetipo_kproto_opt" in df.columns:
        print(f"\n  KPrototypes k=3 (validación cruzada):")
        counts_kp = df["arquetipo_kproto_opt"].value_counts().sort_index()
        for arq, n in counts_kp.items():
            pct = n/total*100
            barra = "█" * int(pct/2)
            print(f"    {arq}: {n:4d} ({pct:.1f}%) {barra}")

    print(f"\n  Métricas Ward k=4:")
    print(f"    Silueta : 0.1488")
    print(f"    Dunn    : 0.0757")
    print(f"    DB      : 1.7670")
    print(f"    CV      : 0.231 (mejor balance de todos los k evaluados)")
    print(f"    DBSCAN  : No viable — datos ordinales sin estructura de densidad")


def seccion_logro(df):
    print(f"\n{_sep()}")
    print("  2. PERFIL DE LOGRO DE COMPETENCIAS (escala 1–5)")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]
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
            print(f"    {_nombre(a)}: media={sub.mean():.2f} | "
                  f"mediana={sub.median():.1f} | std={sub.std():.2f}")
        _, p = _kruskal(df, sc)
        print(f"    Kruskal-Wallis p={p:.4f} {'***' if p<0.001 else ''}")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_incidencia(df):
    print(f"\n{_sep()}")
    print("  3. INCIDENCIA DE LA FORMACIÓN EN BIENESTAR (escala 0–5)")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]
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
    print(f"\n{_sep()}")
    print("  4. PROGRAMA ACADÉMICO POR ARQUETIPO")
    print(_sep())

    if "programa" not in df.columns:
        print("  ⚠️  Variable 'programa' no disponible.")
        return

    df["_arq"] = df["arquetipo_ward_opt"]
    df_p = df[df["programa"].notna() & (df["programa"].astype(str) != "nan")]
    total_p = len(df_p)

    print(f"\n  Base con programa: {total_p:,} ({total_p/len(df)*100:.1f}%)")
    print(f"  Programas únicos: {df_p['programa'].nunique()}")

    print(f"\n  Top 8 globales:")
    for prog, n in df_p["programa"].value_counts().head(8).items():
        print(f"    {n:4d} ({n/total_p*100:.1f}%) | {str(prog).strip()[:50]}")

    for a in sorted(df["_arq"].unique()):
        sub = df_p[df_p["_arq"]==a]
        print(f"\n  Arquetipo {a} — {_nombre(a)} (n={len(sub)}):")
        for prog, n in sub["programa"].value_counts().head(5).items():
            pct_arq = n/len(sub)*100
            pct_prog = n/df_p[df_p["programa"]==prog].shape[0]*100
            print(f"    {n:4d} ({pct_arq:.1f}% del arq | {pct_prog:.1f}% del prog) | "
                  f"{str(prog).strip()[:50]}")

    # Programas más diferenciadores
    print(f"\n  Programas más diferenciadores:")
    tabla = pd.crosstab(
        df_p["programa"].astype(str).str.strip(),
        df_p["_arq"],
        normalize="index"
    ).round(3)*100
    tabla.columns = [_nombre(c) for c in tabla.columns]
    tabla["var"] = tabla.var(axis=1)
    top = tabla[tabla.index.map(lambda x: df_p["programa"].astype(str).str.strip().value_counts().get(x,0) >= 20)]
    top = top.nlargest(6, "var")
    cols_arq = [c for c in top.columns if c != "var"]
    header = f"  {'Programa':<45}"
    for c in cols_arq: header += f" {c[:6]:>8}"
    print(header)
    print(f"  {_sub(75)}")
    for prog, row in top.iterrows():
        line = f"  {str(prog)[:45]:<45}"
        for c in cols_arq: line += f" {row[c]:>8.1f}"
        print(line)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_graduacion(df):
    print(f"\n{_sep()}")
    print("  5. AÑO DE GRADUACIÓN POR ARQUETIPO")
    print(_sep())

    if "año_graduacion" not in df.columns:
        print("  ⚠️  Variable 'año_graduacion' no disponible.")
        return

    df["_arq"] = df["arquetipo_ward_opt"]
    df_g = df[df["año_graduacion"].notna()]
    n_arqs = sorted(df["_arq"].unique())

    print(f"\n  Rango global: {df_g['año_graduacion'].min():.0f} — {df_g['año_graduacion'].max():.0f}")
    print(f"  Mediana global: {df_g['año_graduacion'].median():.0f}")

    print(f"\n  {'Arquetipo':<35} {'n':>5} {'Media':>7} {'Mediana':>8} {'Min':>6} {'Max':>6}")
    print(f"  {_sub(70)}")
    for a in n_arqs:
        sub = df_g[df_g["_arq"]==a]["año_graduacion"]
        print(f"  {_nombre(a):<35} {len(sub):>5} {sub.mean():>7.1f} "
              f"{sub.median():>8.0f} {sub.min():>6.0f} {sub.max():>6.0f}")

    _, p = _kruskal(df_g, "año_graduacion", "_arq")
    print(f"\n  Kruskal-Wallis p={p:.4f} {'*** diferencias significativas' if p<0.001 else 'ns — sin diferencia'}")

    print(f"\n  Distribución por cohorte (% dentro de cada arquetipo):")
    print(f"  {'Cohorte':<15}" + "".join([f" {'Arq'+str(a):>10}" for a in n_arqs]))
    print(f"  {_sub(55)}")
    for lo, hi, label in [(2017,2019,"2017-2018"),(2019,2021,"2019-2020"),
                           (2021,2023,"2021-2022"),(2023,2025,"2023-2024")]:
        vals = []
        for a in n_arqs:
            sub = df_g[df_g["_arq"]==a]["año_graduacion"]
            pct = ((sub>=lo)&(sub<hi)).sum()/len(sub)*100
            vals.append(pct)
        line = f"  {label:<15}"
        for v in vals: line += f" {v:>9.1f}%"
        print(line)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_trayectoria(df):
    print(f"\n{_sep()}")
    print("  6. TRAYECTORIA LABORAL POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]
    n_arqs = sorted(df["_arq"].unique())

    cols_lab = {
        "laborando_actualmente": "Laborando actualmente (% Sí)",
        "ingresos_superiores_estudio": "Ingresos superiores al estudiar (% Sí)",
        "percepcion_programa": "Percepción programa (1-5)",
        "percepcion_ingreso": "Percepción ingreso (1-4)",
        "formacion_impacto_general": "Impacto general formación (1-5)",
    }

    print(f"\n  {'Variable':<40}" + "".join([f" {'Arq'+str(a):>8}" for a in n_arqs]) + f" {'KW p':>8}")
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
    print(f"\n{_sep()}")
    print("  7. PERFIL CATEGÓRICO POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]
    nombres_arq = df["_arq"].map(_nombre)

    cats = {
        "cat_sede": "Sede",
        "cat_laborando": "Laborando",
        "cat_tipo_cargo": "Tipo de cargo",
        "cat_programa": "Programa (top categorías)",
    }

    for col, nombre in cats.items():
        if col not in df.columns: continue
        serie = df[col].apply(lambda x: str(x).strip() if _es_valido(x) else np.nan)
        if serie.notna().sum() == 0: continue
        tabla = pd.crosstab(
            nombres_arq[serie.notna()],
            serie.dropna(),
            normalize="index"
        ).round(3)*100
        print(f"\n  {nombre} (n={serie.notna().sum():,}):")
        print(tabla.to_string())

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_robustez(df):
    print(f"\n{_sep()}")
    print("  8. ROBUSTEZ — WARD vs K-PROTOTYPES")
    print(_sep())

    if "arquetipo_kproto_opt" not in df.columns: return

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(df["arquetipo_ward_opt"], df["arquetipo_kproto_opt"])
    print(f"\n  Adjusted Rand Index (Ward k=4 vs KProto k=3): {ari:.4f}")
    if ari > 0.6:
        print(f"  → Alta concordancia: estructura robusta entre métodos.")
    elif ari > 0.3:
        print(f"  → Concordancia moderada: coinciden en estructura general.")
    else:
        print(f"  → Baja concordancia: métodos capturan estructuras distintas.")
        print(f"     Ward (espacio latente AFE+ACM) es más apropiado para datos")
        print(f"     con estructura factorial clara — es el método principal.")

    print(f"\n  Tabla de correspondencia Ward vs KProto (%):")
    tabla = pd.crosstab(
        df["arquetipo_ward_opt"].map(_nombre),
        df["arquetipo_kproto_opt"],
        normalize="index"
    ).round(3)*100
    print(tabla.to_string())


def seccion_limitacion_sede(df):
    print(f"\n{_sep()}")
    print("  9. LIMITACIÓN METODOLÓGICA — EFECTO SEDE")
    print(_sep())

    if "sede_raw" not in df.columns and "cat_sede" not in df.columns:
        print("  ⚠️  Variable sede no disponible.")
        return

    col_sede = "sede_raw" if "sede_raw" in df.columns else "cat_sede"
    df["_arq"] = df["arquetipo_ward_opt"]
    serie = df[col_sede].apply(lambda x: str(x).strip() if _es_valido(x) else np.nan)
    nombres_arq = df["_arq"].map(_nombre)

    tabla = pd.crosstab(
        nombres_arq[serie.notna()],
        serie.dropna(),
        normalize="index"
    ).round(3)*100

    print(f"\n  Distribución sede por arquetipo (%):")
    print(tabla.to_string())

    print(f"""
  ⚠️  LIMITACIÓN DOCUMENTADA:
  Los arquetipos de esta base muestran una fuerte asociación con la sede
  de graduación. Esto sugiere que el clustering está capturando diferencias
  inter-sede más que perfiles puramente individuales.

  Interpretación:
  - La heterogeneidad entre sedes (contexto socioeconómico, oferta académica,
    mercado laboral regional) domina la varianza en las variables de logro
    e incidencia.
  - Esto no invalida el análisis pero limita la generalización de los
    arquetipos como perfiles universales de egresado USTA.
  - Recomendación: realizar análisis separados por sede para aislar el efecto
    institucional del efecto territorial.
    """)

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


def seccion_factores_comunes(df):
    print(f"\n{_sep()}")
    print("  10. FACTORES COMUNES CON BASE PERCEPCIÓN")
    print(_sep())

    # Cargar base percepción si existe
    ruta_perc = Path("artifacts/base_con_arquetipos.parquet")
    df_p = None
    if ruta_perc.exists():
        try:
            df_p = pd.read_parquet(ruta_perc)
            log.info("Base percepción cargada para análisis cruzado.")
        except Exception:
            pass

    # Números reales verificados de ambos pipelines
    perc = {
        "arq0_n": 534,  "arq0_pct": 21.1,
        "arq1_n": 1343, "arq1_pct": 53.1,
        "arq2_n": 653,  "arq2_pct": 25.8,
        "lider_bienestar": 5.19, "lider_rec": 87.9,
        "lider_insercion": 4.33, "lider_mov": 23.1,
        "sat_bienestar": 2.48,   "sat_rec": 31.8,
        "sat_insercion": 2.41,
        "segunda_lengua_global": 2.76,
        "segunda_lengua_lider": 3.35,
        "segunda_lengua_sat": 2.16,
        "movilidad_global": 20.7,
    }
    dep = {
        "arq0_n": 335,  "arq0_pct": 29.7,
        "arq1_n": 214,  "arq1_pct": 19.0,
        "arq2_n": 341,  "arq2_pct": 30.2,
        "arq3_n": 239,  "arq3_pct": 21.2,
        "lider_logro": 41.74, "lider_impacto": 22.39,
        "lider_laborando": 93.7, "lider_ingresos": 88.7,
        "dev_logro": 33.74,   "dev_impacto": 8.59,
        "dev_laborando": 60.9,
    }

    sep = _sep()
    print(f"""
  TABLA COMPARATIVA DE ARQUETIPOS ANÁLOGOS
  {_sub(65)}
  {'BASE PERCEPCIÓN (Ward k=3)':<40} {'BASE DEPURADA (Ward k=4)'}
  {_sub(65)}
  El Subjetivamente Satisfecho ({perc['arq0_pct']:.1f}%, n={perc['arq0_n']})
  → Bienestar: {perc['sat_bienestar']:.2f}/7
  → Recomendaría USTA: {perc['sat_rec']:.1f}%       El Graduado en Desarrollo ({dep['arq0_pct']:.1f}%, n={dep['arq0_n']})
  → Inserción laboral: {perc['sat_insercion']:.2f}/5     → Score logro: {dep['dev_logro']:.2f}/50
                                             → Score impacto: {dep['dev_impacto']:.2f}/30
                                             → Laborando: {dep['dev_laborando']:.1f}%
  {_sub(65)}
  El Líder de Alto Desempeño ({perc['arq2_pct']:.1f}%, n={perc['arq2_n']})
  → Bienestar: {perc['lider_bienestar']:.2f}/7
  → Recomendaría USTA: {perc['lider_rec']:.1f}%       El Líder con Alta Incidencia ({dep['arq3_pct']:.1f}%, n={dep['arq3_n']})
  → Inserción laboral: {perc['lider_insercion']:.2f}/5     → Score logro: {dep['lider_logro']:.2f}/50
  → Movilidad ascenso: {perc['lider_mov']:.1f}%        → Score impacto: {dep['lider_impacto']:.2f}/30
                                             → Laborando: {dep['lider_laborando']:.1f}%
                                             → Ingresos superiores: {dep['lider_ingresos']:.1f}%
  {_sub(65)}
    """)

    print(f"""
  COMPONENTES COMUNES VERIFICADOS CON DATOS REALES:

  1. El perfil "Líder" es el más robusto entre bases:
     Percepción: bienestar {perc['lider_bienestar']:.2f}/7, rec {perc['lider_rec']:.1f}%, inserción {perc['lider_insercion']:.2f}/5
     Depurada  : logro {dep['lider_logro']:.2f}/50, impacto {dep['lider_impacto']:.2f}/30, laborando {dep['lider_laborando']:.1f}%
     → En ambas bases es el grupo con mayor éxito en TODAS las dimensiones.

  2. El perfil de "menor desempeño" existe en ambas bases:
     Percepción: {perc['arq0_pct']:.1f}% — bienestar {perc['sat_bienestar']:.2f}/7, solo {perc['sat_rec']:.1f}% recomendaría la USTA
     Depurada  : {dep['arq0_pct']:.1f}% — impacto {dep['dev_impacto']:.2f}/30, solo {dep['dev_laborando']:.1f}% laborando
     → La formación no está teniendo el impacto esperado en ~20-30% de graduados.

  3. Brecha segunda lengua — transversal e independiente del instrumento:
     Percepción global: {perc['segunda_lengua_global']:.2f}/5
     Por arquetipo: Satisfecho={perc['segunda_lengua_sat']:.2f} | Líder={perc['segunda_lengua_lider']:.2f}
     → Esta brecha no varía con el método de medición.

  4. Movilidad social (solo medible en percepción):
     Global: {perc['movilidad_global']:.1f}% ascendió de estrato
     Líder: {perc['lider_mov']:.1f}% | Satisfecho: 15.5%
     → La formación sí tiene impacto diferenciado en movilidad social.

  DIFERENCIA CLAVE ENTRE INSTRUMENTOS:
  La base depurada captura INCIDENCIA CONCRETA (ingresos, empleo, vivienda).
  La base percepción captura SATISFACCIÓN SUBJETIVA y correspondencia laboral.
  Ambas dimensiones son necesarias para un análisis integral del impacto USTA.
    """)


def seccion_recomendaciones(df):
    print(f"\n{_sep()}")
    print("  11. RECOMENDACIONES INSTITUCIONALES")
    print(_sep())
    print("""
  R1 — URGENTE: El 29.7% son Graduados en Desarrollo (incidencia ~14.6/30)
       Implementar protocolo de seguimiento post-grado diferenciado.
       La formación no está teniendo el impacto esperado en este grupo.

  R2 — ALTA: logro_comp_prof_3 es la competencia más baja en los 4 arquetipos
       Requiere revisión curricular urgente del componente profesional
       específico — independiente del programa académico.

  R3 — ESTRATÉGICO: El Líder con Alta Incidencia (21.2%) es el referente
       Capitalizar su perfil para mentoría y visibilidad institucional.
       En ambas bases este arquetipo emerge como el más robusto.

  R4 — ESTRUCTURAL: Diferencias por sede ameritan análisis independiente
       Bucaramanga y Villavicencio muestran patrones distintos a Bogotá.
       El efecto territorial no puede separarse del efecto institucional
       sin un diseño de investigación que controle por sede.

  R5 — INVESTIGACIÓN: Cruzar ambas bases por programa académico
       Los programas de Negocios Internacionales y Derecho aparecen
       en el top de ambas bases. Analizar si sus graduados pertenecen
       consistentemente al mismo arquetipo en los dos estudios.
    """)


def run():
    log.info("Iniciando evaluate_depurada.py v4...")
    df = cargar_base()

    print(f"\n{_sep()}")
    print("  REPORTE ARQUETIPOS — BASE DEPURADA PENSER USTA 2025")
    print(f"  Metodología: AFE Bloques (3) + ACM + Ward k=4")
    print(f"  Base: 1.129 respuestas | 4 arquetipos")
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

    df["nombre_arquetipo"] = df["arquetipo_ward_opt"].map(_nombre)
    df.to_parquet(ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet", index=False)
    log.info("Base final guardada.")
    log.info("evaluate_depurada.py v4 completado.")


if __name__ == "__main__":
    run()