"""
evaluate.py — Versión 4.0
==========================
Descriptivas profundas por arquetipo — Base Percepción Egresados USTA

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Contenido:
----------
1.  Distribución de arquetipos (Ward k=3 vs KProto k=3)
2.  Perfil de competencias por arquetipo (medias + brechas)
3.  Perfil de bienestar y satisfacción
4.  Carrera/programa por arquetipo
5.  Edad por arquetipo — hipótesis del líder más antiguo
6.  Año de graduación por arquetipo
7.  Trayectoria laboral (cargo, salario, sector, tiempo primer empleo)
8.  Movilidad social por arquetipo
9.  Perfil categórico (género, sede, estado civil, nivel educativo padres)
10. Comparación Ward vs KProto — robustez de la solución
11. Recomendaciones institucionales con evidencia estadística
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

# ---------------------------------------------------------------------------
# Nombres de arquetipos Ward k=3
# ---------------------------------------------------------------------------
ARQUETIPOS = {
    0: {
        "nombre": "El Subjetivamente Satisfecho",
        "color": "🔴",
        "descripcion": (
            "Graduado con el menor logro en competencias y menor bienestar percibido. "
            "Sus competencias cognitivas (B1), tecnológicas (B2) y de liderazgo (B3) "
            "están en rango bajo. La correspondencia entre su formación y su empleo es "
            "la más baja del grupo. Solo el 31.8% recomendaría la USTA."
        ),
    },
    1: {
        "nombre": "El Profesional Consolidado",
        "color": "🟡",
        "descripcion": (
            "Perfil mayoritario (53.1%). Competencias medias-altas, satisfacción "
            "moderada con la formación y buena inserción laboral. Representa al "
            "graduado que aprovechó la formación para insertarse exitosamente. "
            "El 90.5% recomendaría la USTA."
        ),
    },
    2: {
        "nombre": "El Líder de Alto Desempeño",
        "color": "🟢",
        "descripcion": (
            "El perfil de mayor éxito institucional (25.8%). Competencias muy altas "
            "en los 6 bloques, bienestar material y social elevado, alta satisfacción "
            "y correspondencia laboral. El 84.4% recomendaría la USTA. "
            "Referente del modelo educativo USTA."
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
    "acceso_seguridad_social", "incremento_cultural", "satisfecho_ocio", "red_amigos",
]

COLS_SATISFACCION = [
    "satisfaccion_formacion", "efecto_calidad_vida", "satisfaccion_vida",
    "correspondencia_primer_empleo", "correspondencia_empleo_actual",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def _sep(n=65):
    return "=" * n


def _subsep(n=65):
    return "─" * n


def _kruskal_test(df, col, arq_col="arquetipo"):
    """Test de Kruskal-Wallis para diferencias entre arquetipos."""
    grupos = [_num(df[df[arq_col] == a], col).dropna()
              for a in sorted(df[arq_col].unique())]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2:
        return np.nan, np.nan
    try:
        stat, p = stats.kruskal(*grupos)
        return round(stat, 3), round(p, 4)
    except Exception:
        return np.nan, np.nan


def _nombre_arq(x):
    return ARQUETIPOS.get(int(x), {}).get("nombre", f"Arquetipo {x}")


# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------

def cargar_base():
    ruta = ARTIFACTS_PATH / "base_con_arquetipos.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No encontrado: {ruta}\nEjecuta: python src/percepcion/train.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} × {df.shape[1]}")

    # Reconstruir variables demográficas desde raw si están disponibles
    raw = Path("data/raw/ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx")
    if raw.exists():
        df_raw = pd.read_excel(raw)
        df_raw = df_raw.drop(columns=["Unnamed: 0"], errors="ignore")
        df_raw = df_raw[~df_raw.isnull().all(axis=1)]
        df_raw = df_raw.drop(
            columns=df_raw.columns[df_raw.isnull().mean() > 0.95].tolist(),
            errors="ignore"
        )
        col_ff = '          Exponer las ideas de forma clara y efectiva por medios   escritos\n        '
        df_raw = df_raw[df_raw[col_ff] != 87.0]
        df_raw = df_raw.drop_duplicates().reset_index(drop=True)

        # Edad
        col_nac = "Fecha de nacimiento:"
        if col_nac in df_raw.columns:
            fechas = pd.to_datetime(df_raw[col_nac], errors="coerce")
            edad = (pd.Timestamp("2026-01-01") - fechas).dt.days / 365.25
            df["edad"] = edad.where((edad >= 18) & (edad <= 80)).values[:len(df)]

        # Año graduación
        col_grad = [c for c in df_raw.columns if "FECHA DE GRADUACIÓN" in c]
        if col_grad:
            años = pd.to_datetime(df_raw[col_grad[0]], errors="coerce").dt.year
            df["año_graduacion"] = años.where((años >= 1970) & (años <= 2026)).values[:len(df)]
            df["años_graduado"] = (2026 - df["año_graduacion"]).where(df["año_graduacion"].notna())

        # Programa académico
        prog_cols = [c for c in df_raw.columns
                     if "Programa del pregrado" in c or "Programa del posgrado" in c]
        if prog_cols:
            prog = pd.Series([np.nan] * len(df_raw))
            for col in prog_cols:
                prog = prog.fillna(df_raw[col])
            df["programa"] = prog.values[:len(df)]

        log.info("Variables demográficas reconstruidas desde raw.")

    return df


# ---------------------------------------------------------------------------
# 1. Distribución
# ---------------------------------------------------------------------------

def seccion_distribucion(df):
    print(f"\n{_sep()}")
    print("  1. DISTRIBUCIÓN DE ARQUETIPOS")
    print(_sep())
    total = len(df)

    print(f"\n  Ward k=3 (método principal):")
    counts = df["arquetipo_ward_opt"].value_counts().sort_index()
    for arq, n in counts.items():
        nombre = _nombre_arq(arq)
        pct = n / total * 100
        barra = "█" * int(pct / 2)
        print(f"    {arq} — {nombre:<38}: {n:4d} ({pct:.1f}%) {barra}")

    if "arquetipo_kproto_opt" in df.columns:
        print(f"\n  KPrototypes k=3 (validación cruzada):")
        counts_kp = df["arquetipo_kproto_opt"].value_counts().sort_index()
        for arq, n in counts_kp.items():
            pct = n / total * 100
            barra = "█" * int(pct / 2)
            print(f"    {arq}: {n:4d} ({pct:.1f}%) {barra}")

    print(f"\n  Métricas de validación (Ward k=3):")
    print(f"    Silueta : 0.1029 (separación moderada — esperable en datos Likert)")
    print(f"    Dunn    : 0.1095 (mayor = mayor separación inter-cluster)")
    print(f"    DB      : 2.1836 (menor = mejor — valor aceptable)")
    print(f"    CV      : 0.518  (balance 21.1% / 53.1% / 25.8%)")
    print(f"    DBSCAN  : No viable — datos Likert sin estructura de densidad")


# ---------------------------------------------------------------------------
# 2. Competencias por arquetipo
# ---------------------------------------------------------------------------

def seccion_competencias(df):
    print(f"\n{_sep()}")
    print("  2. PERFIL DE COMPETENCIAS POR ARQUETIPO (escala 1–5)")
    print(_sep())

    cols = [c for c in COLS_COMPETENCIAS if c in df.columns]
    df["_arq"] = df["arquetipo_ward_opt"]

    perfil = df.groupby("_arq")[cols].mean().round(3)
    perfil.index = perfil.index.map(_nombre_arq)

    print(f"\n  {'Competencia':<35} {'Arq 0':>8} {'Arq 1':>8} {'Arq 2':>8} {'KW p':>8}")
    print(f"  {_subsep()}")
    for col in cols:
        medias = [_num(df[df["_arq"] == a], col).mean() for a in range(3)]
        _, p = _kruskal_test(df, col, "_arq")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<35} {medias[0]:>8.3f} {medias[1]:>8.3f} {medias[2]:>8.3f} {p:>7.4f}{sig}")

    print(f"\n  SEGUNDA LENGUA — Brecha transversal:")
    print(f"    Global: {_num(df, 'segunda_lengua').mean():.2f}/5")
    for a in range(3):
        m = _num(df[df["_arq"] == a], "segunda_lengua").mean()
        print(f"    {_nombre_arq(a)}: {m:.2f}/5")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 3. Bienestar y satisfacción
# ---------------------------------------------------------------------------

def seccion_bienestar(df):
    print(f"\n{_sep()}")
    print("  3. BIENESTAR Y SATISFACCIÓN POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]

    print(f"\n  Score bienestar (0–7):")
    for a in range(3):
        sub = _num(df[df["_arq"] == a], "score_bienestar")
        print(f"    {_nombre_arq(a)}: media={sub.mean():.2f} | mediana={sub.median():.1f} | std={sub.std():.2f}")
    _, p = _kruskal_test(df, "score_bienestar", "_arq")
    print(f"    Kruskal-Wallis p={p:.4f} {'***' if p<0.001 else ''}")

    print(f"\n  Indicadores de bienestar (% respondió Sí):")
    cols_b = [c for c in COLS_BIENESTAR if c in df.columns]
    for col in cols_b:
        vals = [_num(df[df["_arq"] == a], col).mean() * 100 for a in range(3)]
        print(f"    {col:<35}: {vals[0]:>6.1f}% | {vals[1]:>6.1f}% | {vals[2]:>6.1f}%")

    print(f"\n  Satisfacción (escala 0–5):")
    cols_s = [c for c in COLS_SATISFACCION if c in df.columns]
    for col in cols_s:
        vals = [_num(df[df["_arq"] == a], col).mean() for a in range(3)]
        _, p = _kruskal_test(df, col, "_arq")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {col:<40}: {vals[0]:>5.2f} | {vals[1]:>5.2f} | {vals[2]:>5.2f}  p={p:.4f}{sig}")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 4. Programa académico
# ---------------------------------------------------------------------------

def seccion_programa(df):
    print(f"\n{_sep()}")
    print("  4. PROGRAMA ACADÉMICO POR ARQUETIPO")
    print(_sep())

    if "programa" not in df.columns:
        print("  ⚠️  Variable 'programa' no disponible.")
        return

    df["_arq"] = df["arquetipo_ward_opt"]
    df_prog = df[df["programa"].notna() & (df["programa"] != "nan")]
    total_prog = len(df_prog)

    print(f"\n  Base con programa disponible: {total_prog:,} ({total_prog/len(df)*100:.1f}%)")
    print(f"  Programas únicos: {df_prog['programa'].nunique()}")

    print(f"\n  Top 8 programas globales:")
    for prog, n in df_prog["programa"].value_counts().head(8).items():
        print(f"    {n:4d} ({n/total_prog*100:.1f}%) | {str(prog)[:55]}")

    for a in range(3):
        sub = df_prog[df_prog["_arq"] == a]
        nombre = _nombre_arq(a)
        print(f"\n  Arquetipo {a} — {nombre} (n={len(sub)} con programa):")
        top5 = sub["programa"].value_counts().head(5)
        for prog, n in top5.items():
            pct_arq = n / len(sub) * 100
            pct_prog = n / df_prog[df_prog["programa"] == prog].shape[0] * 100
            print(f"    {n:4d} ({pct_arq:.1f}% del arq | {pct_prog:.1f}% del programa) | {str(prog)[:50]}")

    # Programa más diferenciador
    print(f"\n  Programas más diferenciadores entre arquetipos:")
    tabla = pd.crosstab(
        df_prog["programa"], df_prog["_arq"],
        normalize="index"
    ).round(3) * 100
    tabla.columns = [_nombre_arq(c) for c in tabla.columns]

    # Calcular varianza entre columnas para encontrar los más diferenciadores
    tabla["varianza"] = tabla.var(axis=1)
    top_dif = tabla.nlargest(8, "varianza")
    print(f"\n  {'Programa':<45} {'Arq0%':>7} {'Arq1%':>7} {'Arq2%':>7}")
    print(f"  {_subsep(70)}")
    for prog, row in top_dif.iterrows():
        cols_arq = [c for c in row.index if c != "varianza"]
        vals = [row[c] for c in cols_arq]
        print(f"  {str(prog)[:45]:<45} {vals[0]:>7.1f} {vals[1]:>7.1f} {vals[2]:>7.1f}")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 5. Edad por arquetipo — hipótesis del líder más antiguo
# ---------------------------------------------------------------------------

def seccion_edad(df):
    print(f"\n{_sep()}")
    print("  5. EDAD POR ARQUETIPO")
    print(f"     Hipótesis: '¿El Líder de Alto Desempeño son los más antiguos?'")
    print(_sep())

    if "edad" not in df.columns:
        print("  ⚠️  Variable 'edad' no disponible.")
        return

    df["_arq"] = df["arquetipo_ward_opt"]
    edad_valida = df[df["edad"].notna()]

    print(f"\n  Edad global (n={len(edad_valida):,}):")
    print(f"    Media: {edad_valida['edad'].mean():.1f} años")
    print(f"    Mediana: {edad_valida['edad'].median():.1f} años")
    print(f"    Rango: {edad_valida['edad'].min():.0f} — {edad_valida['edad'].max():.0f} años")

    print(f"\n  Edad por arquetipo:")
    print(f"  {'Arquetipo':<40} {'n':>5} {'Media':>7} {'Mediana':>8} {'Std':>6} {'Min':>6} {'Max':>6}")
    print(f"  {_subsep(75)}")
    for a in range(3):
        sub = edad_valida[edad_valida["_arq"] == a]["edad"]
        print(f"  {_nombre_arq(a):<40} {len(sub):>5} {sub.mean():>7.1f} {sub.median():>8.1f} "
              f"{sub.std():>6.1f} {sub.min():>6.0f} {sub.max():>6.0f}")

    # Test estadístico
    grupos = [edad_valida[edad_valida["_arq"] == a]["edad"].dropna() for a in range(3)]
    stat, p = stats.kruskal(*grupos)
    print(f"\n  Kruskal-Wallis: H={stat:.3f}, p={p:.4f} {'*** diferencias significativas' if p<0.001 else ''}")

    # Distribución por rangos de edad
    print(f"\n  Distribución por rangos de edad (% dentro de cada arquetipo):")
    print(f"  {'Rango':<15} {'Arq 0':>10} {'Arq 1':>10} {'Arq 2':>10}")
    print(f"  {_subsep(50)}")
    rangos = [(18,30,"18-29"),(30,40,"30-39"),(40,50,"40-49"),(50,100,"50+")]
    for lo, hi, label in rangos:
        vals = []
        for a in range(3):
            sub = edad_valida[edad_valida["_arq"] == a]["edad"]
            pct = ((sub >= lo) & (sub < hi)).sum() / len(sub) * 100
            vals.append(pct)
        print(f"  {label:<15} {vals[0]:>9.1f}% {vals[1]:>9.1f}% {vals[2]:>9.1f}%")

    # Conclusión hipótesis
    medias = [edad_valida[edad_valida["_arq"] == a]["edad"].mean() for a in range(3)]
    lider_edad = medias[2]
    max_edad = max(medias)
    print(f"\n  CONCLUSIÓN HIPÓTESIS:")
    if abs(lider_edad - max_edad) < 1.0:
        print(f"  ✅ CONFIRMADA — El Líder de Alto Desempeño tiene la mayor edad media "
              f"({lider_edad:.1f} años)")
        print(f"     Esto es consistente con la teoría del capital humano: a mayor")
        print(f"     experiencia y tiempo de madurez profesional, mayor capitalización")
        print(f"     de las competencias adquiridas durante la formación.")
    else:
        arq_mayor = int(np.argmax(medias))
        print(f"  ❌ NO CONFIRMADA — El arquetipo con mayor edad es el {arq_mayor} "
              f"({_nombre_arq(arq_mayor)}: {max_edad:.1f} años)")
        print(f"     El Líder tiene {lider_edad:.1f} años de media.")
        print(f"     Factores adicionales explican el alto desempeño más allá de la edad.")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 6. Año de graduación
# ---------------------------------------------------------------------------

def seccion_graduacion(df):
    print(f"\n{_sep()}")
    print("  6. AÑO DE GRADUACIÓN Y TIEMPO COMO GRADUADO")
    print(_sep())

    if "año_graduacion" not in df.columns:
        print("  ⚠️  Variable 'año_graduacion' no disponible.")
        return

    df["_arq"] = df["arquetipo_ward_opt"]
    df_g = df[df["año_graduacion"].notna()]

    print(f"\n  Año de graduación por arquetipo:")
    print(f"  {'Arquetipo':<40} {'n':>5} {'Media':>7} {'Mediana':>8} {'Min':>6} {'Max':>6}")
    print(f"  {_subsep(75)}")
    for a in range(3):
        sub = df_g[df_g["_arq"] == a]["año_graduacion"]
        print(f"  {_nombre_arq(a):<40} {len(sub):>5} {sub.mean():>7.1f} "
              f"{sub.median():>8.0f} {sub.min():>6.0f} {sub.max():>6.0f}")

    print(f"\n  Años transcurridos desde graduación:")
    for a in range(3):
        sub = df_g[df_g["_arq"] == a]["años_graduado"]
        print(f"    {_nombre_arq(a)}: media={sub.mean():.1f} años | mediana={sub.median():.0f} | "
              f"<5a={( sub<5).sum()} | 5-10a={(( sub>=5)&(sub<10)).sum()} | "
              f"10-20a={((sub>=10)&(sub<20)).sum()} | >20a={(sub>=20).sum()}")

    # Cohortes
    print(f"\n  Distribución por cohorte (% dentro de cada arquetipo):")
    print(f"  {'Cohorte':<20} {'Arq 0':>10} {'Arq 1':>10} {'Arq 2':>10}")
    print(f"  {_subsep(55)}")
    cohortes = [
        (1970, 2000, "Pre-2000"),
        (2000, 2010, "2000-2009"),
        (2010, 2015, "2010-2014"),
        (2015, 2020, "2015-2019"),
        (2020, 2026, "2020-2022"),
    ]
    for lo, hi, label in cohortes:
        vals = []
        for a in range(3):
            sub = df_g[df_g["_arq"] == a]["año_graduacion"]
            pct = ((sub >= lo) & (sub < hi)).sum() / len(sub) * 100
            vals.append(pct)
        print(f"  {label:<20} {vals[0]:>9.1f}% {vals[1]:>9.1f}% {vals[2]:>9.1f}%")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 7. Trayectoria laboral
# ---------------------------------------------------------------------------

def seccion_trayectoria(df):
    print(f"\n{_sep()}")
    print("  7. TRAYECTORIA LABORAL POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]

    cols_lab = {
        "nivel_cargo_actual": "Nivel de cargo (1=Op. 5=Dir.)",
        "salario_actual": "Salario actual (1=bajo 4=alto)",
        "tiempo_primer_empleo": "Tiempo 1er empleo (1=rápido 4=lento)",
        "movilidad_social": "Movilidad social (estratos)",
    }

    print(f"\n  {'Variable':<35} {'Arq 0':>8} {'Arq 1':>8} {'Arq 2':>8} {'KW p':>8}")
    print(f"  {_subsep()}")
    for col, label in cols_lab.items():
        if col not in df.columns:
            continue
        vals = [_num(df[df["_arq"] == a], col).mean() for a in range(3)]
        _, p = _kruskal_test(df, col, "_arq")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {label:<35} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {p:>7.4f} {sig}")

    # Recomendaría
    print(f"\n  Recomendaría la USTA (%):")
    if "cat_recomendaria" in df.columns:
        for a in range(3):
            sub = df[df["_arq"] == a]["cat_recomendaria"]
            sub = sub[sub.isin(["Si", "No"])]
            pct = (sub == "Si").sum() / len(sub) * 100
            print(f"    {_nombre_arq(a)}: {pct:.1f}% ({(sub=='Si').sum()}/{len(sub)})")

    # Estudiaría otra vez
    print(f"\n  Estudiaría otra vez en la USTA (%):")
    if "cat_estudiaria_otra_vez" in df.columns:
        for a in range(3):
            sub = df[df["_arq"] == a]["cat_estudiaria_otra_vez"]
            sub = sub[sub.isin(["Si", "No", "No lo sabe"])]
            for resp in ["Si", "No", "No lo sabe"]:
                pct = (sub == resp).sum() / len(sub) * 100
                if pct > 0:
                    print(f"    {_nombre_arq(a)} — {resp}: {pct:.1f}%")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 8. Movilidad social
# ---------------------------------------------------------------------------

def seccion_movilidad(df):
    print(f"\n{_sep()}")
    print("  8. MOVILIDAD SOCIAL POR ARQUETIPO")
    print(_sep())

    if "movilidad_social" not in df.columns:
        print("  ⚠️  Variable 'movilidad_social' no disponible.")
        return

    df["_arq"] = df["arquetipo_ward_opt"]
    df_m = df[df["movilidad_social"].notna()]

    print(f"\n  {'Tipo':<15} {'Global':>8} {'Arq 0':>8} {'Arq 1':>8} {'Arq 2':>8}")
    print(f"  {_subsep(55)}")
    mov = _num(df_m, "movilidad_social")
    for label, cond in [
        ("Ascenso", mov > 0),
        ("Sin cambio", mov == 0),
        ("Descenso", mov < 0),
    ]:
        pct_global = cond.sum() / len(mov) * 100
        vals = []
        for a in range(3):
            sub_m = _num(df_m[df_m["_arq"] == a], "movilidad_social")
            if label == "Ascenso":
                pct = (sub_m > 0).sum() / len(sub_m) * 100
            elif label == "Sin cambio":
                pct = (sub_m == 0).sum() / len(sub_m) * 100
            else:
                pct = (sub_m < 0).sum() / len(sub_m) * 100
            vals.append(pct)
        print(f"  {label:<15} {pct_global:>7.1f}% {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%")

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 9. Perfil categórico
# ---------------------------------------------------------------------------

def seccion_categoricas(df):
    print(f"\n{_sep()}")
    print("  9. PERFIL CATEGÓRICO POR ARQUETIPO")
    print(_sep())

    df["_arq"] = df["arquetipo_ward_opt"]
    nombres_arq = df["_arq"].map(_nombre_arq)

    cats = {
        "cat_genero": "Género",
        "cat_sede": "Sede",
        "cat_estado_civil": "Estado civil",
        "cat_nivel_educ_padres": "Nivel educativo padres",
    }

    def _es_valido(x):
        return (pd.notna(x) and str(x).strip() not in ["nan", "None", ""]
                and not str(x).strip().lstrip("-").isdigit())

    for col, nombre in cats.items():
        if col not in df.columns:
            continue
        serie = df[col].apply(lambda x: str(x).strip() if _es_valido(x) else np.nan)
        if serie.notna().sum() == 0:
            continue
        tabla = pd.crosstab(
            nombres_arq[serie.notna()],
            serie.dropna(),
            normalize="index"
        ).round(3) * 100
        print(f"\n  {nombre} (n={serie.notna().sum():,}):")
        print(tabla.to_string())

    df.drop(columns=["_arq"], inplace=True, errors="ignore")


# ---------------------------------------------------------------------------
# 10. Comparación Ward vs KProto
# ---------------------------------------------------------------------------

def seccion_robustez(df):
    print(f"\n{_sep()}")
    print("  10. ROBUSTEZ — COMPARACIÓN WARD vs K-PROTOTYPES")
    print(_sep())

    if "arquetipo_kproto_opt" not in df.columns:
        return

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(df["arquetipo_ward_opt"], df["arquetipo_kproto_opt"])
    print(f"\n  Adjusted Rand Index (Ward k=3 vs KProto k=3): {ari:.4f}")
    print(f"  Interpretación:")
    if ari > 0.6:
        print(f"    ARI={ari:.3f} — Alta concordancia entre métodos.")
        print(f"    Los arquetipos son robustos: ambos métodos identifican")
        print(f"    estructuras similares aunque partan de supuestos distintos.")
    elif ari > 0.3:
        print(f"    ARI={ari:.3f} — Concordancia moderada.")
        print(f"    Los métodos coinciden en la estructura general pero difieren")
        print(f"    en la asignación de casos frontera.")
    else:
        print(f"    ARI={ari:.3f} — Baja concordancia.")
        print(f"    Ward y KPrototypes identifican estructuras diferentes.")
        print(f"    Ward (espacio latente AFE+ACM) captura mejor la estructura")
        print(f"    factorial; KPrototypes trabaja directamente en espacio original.")

    # Tabla de confusión Ward vs KProto
    print(f"\n  Tabla de correspondencia Ward vs KProto (%):")
    tabla = pd.crosstab(
        df["arquetipo_ward_opt"].map(_nombre_arq),
        df["arquetipo_kproto_opt"],
        normalize="index"
    ).round(3) * 100
    print(tabla.to_string())


# ---------------------------------------------------------------------------
# 11. Recomendaciones
# ---------------------------------------------------------------------------

def seccion_recomendaciones(df):
    print(f"\n{_sep()}")
    print("  11. RECOMENDACIONES INSTITUCIONALES")
    print(_sep())
    print("""
  R1 — URGENTE: Segunda lengua [evidencia: media global 2.76/5, 43% calificó 1 o 2]
       Implementar política institucional obligatoria de segunda lengua.
       No es suficiente una materia optativa. La brecha es transversal
       a los 3 arquetipos — no es un problema de un perfil sino sistémico.

  R2 — ALTA: Inserción laboral y herramientas modernas [media: 3.42 y 3.47/5]
       Segunda brecha transversal. Fortalecer empleabilidad, prácticas
       obligatorias y actualización en herramientas digitales del sector.

  R3 — URGENTE: Seguimiento diferenciado del Arquetipo 0 [31.8% recomendaría]
       El Subjetivamente Satisfecho (21.1%) no recomienda la USTA y no
       volvería a estudiar aquí. Requiere protocolo de intervención
       post-grado diferenciado — no estrategias institucionales genéricas.

  R4 — ESTRATÉGICO: Programa de mentoría entre arquetipos
       El Líder de Alto Desempeño (25.8%) tiene competencias muy altas.
       Diseñar programa formal de mentoría para transferir buenas prácticas
       hacia los arquetipos 0 y 1 — especialmente en inserción laboral.

  R5 — ESTRUCTURAL: Revisión por programa académico
       Los programas de Ingeniería y Optometría concentran los mejores
       perfiles. Los programas con mayor proporción de Arquetipo 0 requieren
       revisión curricular enfocada en pertinencia con el mercado laboral.

  R6 — INVESTIGACIÓN: Profundizar diferencias por sede
       Bogotá concentra el Arquetipo 0 (58.1%). Bucaramanga tiene mayor
       proporción de Líderes. Esto amerita un estudio de sede específico
       que controle por programa y cohorte de graduación.
    """)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run():
    log.info("Iniciando evaluate.py v4...")
    df = cargar_base()

    sep = _sep()
    print(f"\n{sep}")
    print("  REPORTE DE ARQUETIPOS — ESTUDIO PENSER EGRESADOS USTA")
    print(f"  Metodología: AFE Bloques + ACM + Ward k=3 / KPrototypes k=3")
    print(f"  Base: 2.530 respuestas | {len(df['arquetipo_ward_opt'].unique())} arquetipos")
    print(sep)

    seccion_distribucion(df)
    seccion_competencias(df)
    seccion_bienestar(df)
    seccion_programa(df)
    seccion_edad(df)
    seccion_graduacion(df)
    seccion_trayectoria(df)
    seccion_movilidad(df)
    seccion_categoricas(df)
    seccion_robustez(df)
    seccion_recomendaciones(df)

    # Guardar perfil completo
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    df["nombre_arquetipo"] = df["arquetipo_ward_opt"].map(_nombre_arq)
    df.to_parquet(ARTIFACTS_PATH / "base_con_arquetipos.parquet", index=False)
    log.info("Base final guardada.")
    log.info("evaluate.py v4 completado.")


if __name__ == "__main__":
    run()