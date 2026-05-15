"""
evaluate.py — Versión 5.0
==========================
Descriptivas profundas por arquetipo — Base Percepción Egresados USTA
Metodología: AFE Bloques + ACM + QuantileTransformer + PCA + KMeans k=3
 
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
 
ARQUETIPOS = {
    0: {
        "nombre": "El Profesional en Desarrollo",
        "color": "🟡",
        "descripcion": (
            "Perfil mayoritario (66.2%, n=1.676). Competencias medias, bienestar 3.66/7. "
            "Inserción laboral 3.11/5 — el punto más débil. "
            "El 67.2% recomendaría la USTA."
        ),
    },
    1: {
        "nombre": "El Líder de Alto Desempeño",
        "color": "🟢",
        "descripcion": (
            "El perfil de mayor éxito (16.8%, n=426). Competencias muy altas — "
            "toma_decisiones 4.97, trabajo_equipo 4.96. "
            "Segunda lengua 3.72/5 — la más alta. "
            "Inserción laboral 4.63/5. El 86.6% recomendaría la USTA."
        ),
    },
    2: {
        "nombre": "El Profesional Consolidado",
        "color": "🔵",
        "descripcion": (
            "Perfil de alto bienestar (16.9%, n=428). Competencias altas, "
            "bienestar 4.71/7 — el mayor de los tres grupos. "
            "El 82.0% recomendaría la USTA."
        ),
    },
}
 
COLS_COMPETENCIAS = [
    "com_escrita","com_oral","pensamiento_critico","metodos_cuantitativos",
    "metodos_cualitativos","lectura_academica","argumentacion","segunda_lengua",
    "creatividad","resolucion_conflictos","liderazgo","toma_decisiones",
    "resolucion_problemas","investigacion","herramientas_informaticas",
    "contextos_multiculturales","insercion_laboral","herramientas_modernas",
    "gestion_informacion","trabajo_equipo","aprendizaje_autonomo",
    "conocimientos_multidisciplinares","etica",
]
 
COLS_BIENESTAR = [
    "adquirio_bienes","mejoro_vivienda","mejoro_salud",
    "acceso_seguridad_social","incremento_cultural","satisfecho_ocio","red_amigos",
]
 
COLS_SATISFACCION = [
    "satisfaccion_formacion","efecto_calidad_vida","satisfaccion_vida",
    "correspondencia_primer_empleo","correspondencia_empleo_actual",
]
 
 
def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
 
def _sep(n=65): return "=" * n
def _sub(n=65): return "─" * n
 
def _kruskal_test(df, col, arq_col="_arq"):
    grupos = [_num(df[df[arq_col]==a], col).dropna() for a in sorted(df[arq_col].unique())]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2: return np.nan, np.nan
    try:
        stat, p = stats.kruskal(*grupos)
        return round(stat,3), round(p,4)
    except: return np.nan, np.nan
 
def _nombre_arq(x):
    return ARQUETIPOS.get(int(x), {}).get("nombre", f"Arquetipo {x}")
 
def _get_arq_col(df):
    """Usa arquetipo_kmeans_opt si existe, sino arquetipo_ward_opt."""
    if "arquetipo_kmeans_opt" in df.columns:
        return "arquetipo_kmeans_opt"
    return "arquetipo_ward_opt"
 
 
def cargar_base():
    ruta = ARTIFACTS_PATH / "base_con_arquetipos.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No encontrado: {ruta}\nEjecuta: python src/percepcion/train.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} × {df.shape[1]}")
 
    raw = Path("data/raw/ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx")
    if raw.exists():
        df_raw = pd.read_excel(raw)
        df_raw = df_raw.drop(columns=["Unnamed: 0"], errors="ignore")
        df_raw = df_raw[~df_raw.isnull().all(axis=1)]
        df_raw = df_raw.drop(
            columns=df_raw.columns[df_raw.isnull().mean() > 0.95].tolist(), errors="ignore"
        )
        col_ff = '          Exponer las ideas de forma clara y efectiva por medios   escritos\n        '
        df_raw = df_raw[df_raw[col_ff] != 87.0]
        df_raw = df_raw.drop_duplicates().reset_index(drop=True)
 
        col_nac = "Fecha de nacimiento:"
        if col_nac in df_raw.columns:
            fechas = pd.to_datetime(df_raw[col_nac], errors="coerce")
            edad = (pd.Timestamp("2026-01-01") - fechas).dt.days / 365.25
            df["edad"] = edad.where((edad >= 18) & (edad <= 80)).values[:len(df)]
 
        col_grad = [c for c in df_raw.columns if "FECHA DE GRADUACIÓN" in c]
        if col_grad:
            años = pd.to_datetime(df_raw[col_grad[0]], errors="coerce").dt.year
            df["año_graduacion"] = años.where((años >= 1970) & (años <= 2026)).values[:len(df)]
            df["años_graduado"] = (2026 - df["año_graduacion"]).where(df["año_graduacion"].notna())
 
        prog_cols = [c for c in df_raw.columns
                     if "Programa del pregrado" in c or "Programa del posgrado" in c]
        if prog_cols:
            prog = pd.Series([np.nan]*len(df_raw))
            for col in prog_cols:
                prog = prog.fillna(df_raw[col])
            df["programa"] = prog.values[:len(df)]
 
        log.info("Variables demográficas reconstruidas desde raw.")
    return df
 
 
def seccion_distribucion(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  1. DISTRIBUCIÓN DE ARQUETIPOS")
    print(_sep())
    total = len(df)
 
    print(f"\n  KMeans k=3 (método principal — silueta=0.498):")
    counts = df[arq_col].value_counts().sort_index()
    for arq, n in counts.items():
        nombre = _nombre_arq(arq)
        pct = n/total*100
        barra = "█" * int(pct/2)
        print(f"    {arq} — {nombre:<38}: {n:4d} ({pct:.1f}%) {barra}")
 
    if "arquetipo_ward_opt" in df.columns:
        print(f"\n  Ward k=3 (validación — silueta=0.479):")
        counts_w = df["arquetipo_ward_opt"].value_counts().sort_index()
        for arq, n in counts_w.items():
            pct = n/total*100
            print(f"    {arq}: {n:4d} ({pct:.1f}%)")
 
    print(f"\n  Métricas KMeans k=3:")
    print(f"    Silueta : 0.4980 ✅")
    print(f"    Dunn    : 0.0146")
    print(f"    DB      : 0.7754 ✅")
    print(f"    PCA     : 3 componentes → 73.8% varianza explicada")
    print(f"    Transf. : QuantileTransformer(normal) sobre espacio AFE+ACM")
 
 
def seccion_competencias(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  2. PERFIL DE COMPETENCIAS POR ARQUETIPO (escala 1–5)")
    print(_sep())
 
    cols = [c for c in COLS_COMPETENCIAS if c in df.columns]
    df["_arq"] = df[arq_col]
    n_arqs = sorted(df["_arq"].unique())
 
    header = f"  {'Competencia':<35}"
    for a in n_arqs: header += f" {'Arq'+str(a):>8}"
    header += f" {'KW p':>8}"
    print(f"\n{header}")
    print(f"  {_sub()}")
 
    for col in cols:
        vals = [_num(df[df["_arq"]==a], col).mean() for a in n_arqs]
        _, p = _kruskal_test(df, col)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        row = f"  {col:<35}"
        for v in vals: row += f" {v:>8.3f}"
        row += f" {p:>7.4f}{sig}"
        print(row)
 
    print(f"\n  SEGUNDA LENGUA — Brecha transversal:")
    sl_global = _num(df, "segunda_lengua").mean()
    print(f"    Global: {sl_global:.2f}/5")
    for a in n_arqs:
        m = _num(df[df["_arq"]==a], "segunda_lengua").mean()
        print(f"    {_nombre_arq(a)}: {m:.2f}/5")
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_bienestar(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  3. BIENESTAR Y SATISFACCIÓN POR ARQUETIPO")
    print(_sep())
 
    df["_arq"] = df[arq_col]
    n_arqs = sorted(df["_arq"].unique())
 
    print(f"\n  Score bienestar (0–7):")
    for a in n_arqs:
        sub = _num(df[df["_arq"]==a], "score_bienestar")
        print(f"    {_nombre_arq(a)}: media={sub.mean():.2f} | mediana={sub.median():.1f} | std={sub.std():.2f}")
    _, p = _kruskal_test(df, "score_bienestar")
    print(f"    Kruskal-Wallis p={p:.4f} {'***' if p<0.001 else ''}")
 
    print(f"\n  Indicadores de bienestar (% respondió Sí):")
    cols_b = [c for c in COLS_BIENESTAR if c in df.columns]
    for col in cols_b:
        vals = [_num(df[df["_arq"]==a], col).mean()*100 for a in n_arqs]
        row = f"    {col:<35}:"
        for v in vals: row += f" {v:>6.1f}%"
        print(row)
 
    print(f"\n  Satisfacción (escala 0–5):")
    for col in [c for c in COLS_SATISFACCION if c in df.columns]:
        vals = [_num(df[df["_arq"]==a], col).mean() for a in n_arqs]
        _, p = _kruskal_test(df, col)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        row = f"    {col:<40}:"
        for v in vals: row += f" {v:>5.2f}"
        row += f"  p={p:.4f}{sig}"
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
    df_prog = df[df["programa"].notna() & (df["programa"] != "nan")]
    total_prog = len(df_prog)
 
    print(f"\n  Base con programa: {total_prog:,} ({total_prog/len(df)*100:.1f}%)")
    print(f"  Programas únicos: {df_prog['programa'].nunique()}")
 
    print(f"\n  Top 8 programas globales:")
    for prog, n in df_prog["programa"].value_counts().head(8).items():
        print(f"    {n:4d} ({n/total_prog*100:.1f}%) | {str(prog)[:55]}")
 
    for a in sorted(df["_arq"].unique()):
        sub = df_prog[df_prog["_arq"]==a]
        print(f"\n  {_nombre_arq(a)} (n={len(sub)}):")
        for prog, n in sub["programa"].value_counts().head(5).items():
            pct_arq = n/len(sub)*100
            pct_prog = n/df_prog[df_prog["programa"]==prog].shape[0]*100
            print(f"    {n:4d} ({pct_arq:.1f}% del arq | {pct_prog:.1f}% del prog) | {str(prog)[:50]}")
 
    # Marginal inversa
    print(f"\n  Por carrera → ¿en qué arquetipo se concentran? (marginal inversa):")
    top_progs = df_prog["programa"].value_counts().head(8).index
    tabla = pd.crosstab(
        df_prog["programa"], df_prog["_arq"], normalize="index"
    ).round(3)*100
    tabla.columns = [_nombre_arq(c)[:8] for c in tabla.columns]
    top = tabla.loc[tabla.index.isin(top_progs)]
    print(f"\n  {'Programa':<40}" + "".join([f" {c:>10}" for c in top.columns]))
    print(f"  {_sub(75)}")
    for prog, row in top.iterrows():
        line = f"  {str(prog)[:40]:<40}"
        for c in top.columns: line += f" {float(row[c].iloc[0] if hasattr(row[c], "iloc") else row[c]):>9.1f}%"
        print(line)
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_edad(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  5. EDAD POR ARQUETIPO")
    print(f"     Hipótesis: '¿El Líder de Alto Desempeño son los más antiguos?'")
    print(_sep())
 
    if "edad" not in df.columns:
        print("  ⚠️  Variable 'edad' no disponible.")
        return
 
    df["_arq"] = df[arq_col]
    edad_valida = df[df["edad"].notna()]
 
    print(f"\n  Edad global (n={len(edad_valida):,}):")
    print(f"    Media: {edad_valida['edad'].mean():.1f} años")
    print(f"    Mediana: {edad_valida['edad'].median():.1f} años")
 
    print(f"\n  {'Arquetipo':<40} {'n':>5} {'Media':>7} {'Mediana':>8} {'Std':>6}")
    print(f"  {_sub(65)}")
    for a in sorted(df["_arq"].unique()):
        sub = edad_valida[edad_valida["_arq"]==a]["edad"]
        print(f"  {_nombre_arq(a):<40} {len(sub):>5} {sub.mean():>7.1f} {sub.median():>8.1f} {sub.std():>6.1f}")
 
    grupos = [edad_valida[edad_valida["_arq"]==a]["edad"].dropna() for a in sorted(df["_arq"].unique())]
    stat, p = stats.kruskal(*grupos)
    print(f"\n  Kruskal-Wallis: H={stat:.3f}, p={p:.4f}")
 
    medias = [edad_valida[edad_valida["_arq"]==a]["edad"].mean() for a in sorted(df["_arq"].unique())]
    lider_arq = 1  # Arquetipo 1 es el Líder
    lider_edad = medias[lider_arq]
    max_edad = max(medias)
    print(f"\n  CONCLUSIÓN HIPÓTESIS:")
    if p >= 0.05:
        print(f"  ❌ NO CONFIRMADA — p={p:.4f} (no significativo)")
        print(f"     Las diferencias de edad entre arquetipos no son estadísticamente")
        print(f"     significativas. El alto desempeño no depende de la edad.")
    else:
        print(f"  ✅ CONFIRMADA — diferencias significativas (p={p:.4f})")
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_graduacion(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  6. AÑO DE GRADUACIÓN Y TIEMPO COMO GRADUADO")
    print(_sep())
 
    if "año_graduacion" not in df.columns:
        print("  ⚠️  Variable 'año_graduacion' no disponible.")
        return
 
    df["_arq"] = df[arq_col]
    df_g = df[df["año_graduacion"].notna()]
 
    print(f"\n  {'Arquetipo':<40} {'n':>5} {'Media':>7} {'Mediana':>8} {'Min':>6} {'Max':>6}")
    print(f"  {_sub(70)}")
    for a in sorted(df["_arq"].unique()):
        sub = df_g[df_g["_arq"]==a]["año_graduacion"]
        print(f"  {_nombre_arq(a):<40} {len(sub):>5} {sub.mean():>7.1f} {sub.median():>8.0f} {sub.min():>6.0f} {sub.max():>6.0f}")
 
    grupos = [df_g[df_g["_arq"]==a]["año_graduacion"].dropna() for a in sorted(df["_arq"].unique())]
    _, p = stats.kruskal(*grupos)
    print(f"\n  Kruskal-Wallis p={p:.4f} {'*** diferencias significativas' if p<0.001 else 'ns — sin diferencia'}")
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_trayectoria(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  7. TRAYECTORIA LABORAL POR ARQUETIPO")
    print(_sep())
 
    df["_arq"] = df[arq_col]
    n_arqs = sorted(df["_arq"].unique())
 
    cols_lab = {
        "nivel_cargo_actual": "Nivel de cargo (1=Op. 5=Dir.)",
        "salario_actual": "Salario actual (1=bajo 4=alto)",
        "tiempo_primer_empleo": "Tiempo 1er empleo (1=rápido 4=lento)",
        "movilidad_social": "Movilidad social (estratos)",
    }
 
    header = f"  {'Variable':<35}"
    for a in n_arqs: header += f" {'Arq'+str(a):>8}"
    header += f" {'KW p':>8}"
    print(f"\n{header}")
    print(f"  {_sub()}")
 
    for col, label in cols_lab.items():
        if col not in df.columns: continue
        vals = [_num(df[df["_arq"]==a], col).mean() for a in n_arqs]
        _, p = _kruskal_test(df, col)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        row = f"  {label:<35}"
        for v in vals: row += f" {v:>8.3f}"
        row += f" {p:>7.4f} {sig}"
        print(row)
 
    print(f"\n  Recomendaría la USTA (%):")
    if "cat_recomendaria" in df.columns:
        for a in n_arqs:
            sub = df[df["_arq"]==a]["cat_recomendaria"]
            sub = sub[sub.isin(["Si","No"])]
            pct = (sub=="Si").sum()/len(sub)*100
            print(f"    {_nombre_arq(a)}: {pct:.1f}% ({(sub=='Si').sum()}/{len(sub)})")
 
    print(f"\n  Estudiaría otra vez en la USTA (%):")
    if "cat_estudiaria_otra_vez" in df.columns:
        for a in n_arqs:
            sub = df[df["_arq"]==a]["cat_estudiaria_otra_vez"]
            sub = sub[sub.isin(["Si","No","No lo sabe"])]
            for resp in ["Si","No","No lo sabe"]:
                pct = (sub==resp).sum()/len(sub)*100
                if pct > 0:
                    print(f"    {_nombre_arq(a)} — {resp}: {pct:.1f}%")
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_movilidad(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  8. MOVILIDAD SOCIAL POR ARQUETIPO")
    print(_sep())
 
    if "movilidad_social" not in df.columns:
        print("  ⚠️  Variable 'movilidad_social' no disponible.")
        return
 
    df["_arq"] = df[arq_col]
    df_m = df[df["movilidad_social"].notna()]
    mov = _num(df_m, "movilidad_social")
    n_arqs = sorted(df["_arq"].unique())
 
    header = f"  {'Tipo':<15} {'Global':>8}"
    for a in n_arqs: header += f" {'Arq'+str(a):>8}"
    print(f"\n{header}")
    print(f"  {_sub(55)}")
 
    for label, cond_fn in [("Ascenso", lambda x: x>0), ("Sin cambio", lambda x: x==0), ("Descenso", lambda x: x<0)]:
        pct_global = cond_fn(mov).sum()/len(mov)*100
        row = f"  {label:<15} {pct_global:>7.1f}%"
        for a in n_arqs:
            sub_m = _num(df_m[df_m["_arq"]==a], "movilidad_social")
            pct = cond_fn(sub_m).sum()/len(sub_m)*100
            row += f" {pct:>7.1f}%"
        print(row)
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_categoricas(df):
    arq_col = _get_arq_col(df)
    print(f"\n{_sep()}")
    print("  9. PERFIL CATEGÓRICO POR ARQUETIPO")
    print(_sep())
 
    df["_arq"] = df[arq_col]
    nombres_arq = df["_arq"].map(_nombre_arq)
 
    def _es_valido(x):
        return (pd.notna(x) and str(x).strip() not in ["nan","None",""]
                and not str(x).strip().lstrip("-").isdigit())
 
    for col, nombre in [("cat_genero","Género"),("cat_sede","Sede"),
                         ("cat_estado_civil","Estado civil"),
                         ("cat_nivel_educ_padres","Nivel educativo padres")]:
        if col not in df.columns: continue
        serie = df[col].apply(lambda x: str(x).strip() if _es_valido(x) else np.nan)
        if serie.notna().sum() == 0: continue
        tabla = pd.crosstab(nombres_arq[serie.notna()], serie.dropna(), normalize="index").round(3)*100
        print(f"\n  {nombre} (n={serie.notna().sum():,}):")
        print(tabla.to_string())
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
 
def seccion_robustez(df):
    print(f"\n{_sep()}")
    print("  10. ROBUSTEZ — KMeans vs Ward")
    print(_sep())
 
    if "arquetipo_kmeans_opt" not in df.columns or "arquetipo_ward_opt" not in df.columns:
        print("  ⚠️  Columnas de comparación no disponibles.")
        return
 
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(df["arquetipo_kmeans_opt"], df["arquetipo_ward_opt"])
    print(f"\n  Adjusted Rand Index (KMeans k=3 vs Ward k=3): {ari:.4f}")
    if ari > 0.6:
        print(f"  → Alta concordancia: estructura robusta entre métodos.")
    elif ari > 0.3:
        print(f"  → Concordancia moderada: coinciden en estructura general.")
    else:
        print(f"  → Concordancia baja: métodos capturan estructuras distintas.")
        print(f"     KMeans sobre espacio PCA(QuantileNormal) es el método principal.")
 
    print(f"\n  Métricas comparadas:")
    print(f"    KMeans k=3: Silueta=0.4980 | DB=0.7754 | Dunn=0.0146")
    print(f"    Ward   k=3: Silueta=0.4792 | DB=0.8585 | Dunn=0.0554")
 
 
def seccion_recomendaciones(df):
    arq_col = _get_arq_col(df)
    df["_arq"] = df[arq_col]
 
    # Calcular % recomendaría dinámicamente
    rec_pcts = {}
    if "cat_recomendaria" in df.columns:
        for a in sorted(df["_arq"].unique()):
            sub = df[df["_arq"]==a]["cat_recomendaria"]
            sub = sub[sub.isin(["Si","No"])]
            rec_pcts[a] = f"{(sub=='Si').sum()/len(sub)*100:.1f}%"
 
    df.drop(columns=["_arq"], inplace=True, errors="ignore")
 
    print(f"\n{_sep()}")
    print("  11. RECOMENDACIONES INSTITUCIONALES")
    print(_sep())
    print(f"""
  R1 — URGENTE: Segunda lengua [media global 2.76/5, 43% calificó 1 o 2]
       Implementar política institucional obligatoria de segunda lengua.
       No es suficiente una materia optativa. La brecha es transversal
       a los 3 arquetipos — es una limitación institucional sistémica.
 
  R2 — ALTA: Inserción laboral y herramientas modernas [3.11/5 y 3.15/5 en Arq 0]
       Fortalecer empleabilidad, prácticas obligatorias y actualización
       en herramientas digitales del sector.
 
  R3 — URGENTE: Seguimiento diferenciado del Profesional en Desarrollo
       El 66.2% (1.676 graduados) muestra el mayor potencial de mejora.
       Recomendaría USTA: {rec_pcts.get(0, 'N/A')}. Protocolo post-grado diferenciado.
 
  R4 — ESTRATÉGICO: Programa de mentoría — El Líder como referente
       Arq 1 (Líder, 16.8%) tiene competencias muy altas y {rec_pcts.get(1, 'N/A')} recomendaría.
       Diseñar programa de mentoría para transferir buenas prácticas.
 
  R5 — ESTRUCTURAL: Revisión por programa académico
       Las especializaciones se concentran todas en el Profesional en Desarrollo.
       Revisar pertinencia curricular de posgrados con el mercado laboral.
 
  R6 — INVESTIGACIÓN: Análisis separado por sede
       Las sedes muestran distribuciones distintas de arquetipos.
       Requiere diseño de investigación que controle por sede y programa.
    """)
 
 
def run():
    log.info("Iniciando evaluate.py v5...")
    df = cargar_base()
 
    print(f"\n{_sep()}")
    print("  REPORTE DE ARQUETIPOS — ESTUDIO PENSER EGRESADOS USTA")
    print(f"  Metodología: AFE Bloques + ACM + QuantileTransformer + PCA + KMeans k=3")
    print(f"  Base: 2.530 respuestas | 3 arquetipos | Silueta=0.498")
    print(_sep())
 
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
 
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    df["nombre_arquetipo"] = df[_get_arq_col(df)].map(_nombre_arq)
    df.to_parquet(ARTIFACTS_PATH / "base_con_arquetipos.parquet", index=False)
    log.info("Base final guardada.")
    log.info("evaluate.py v5 completado.")
 
 
if __name__ == "__main__":
    run()