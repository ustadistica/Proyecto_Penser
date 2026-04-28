"""
train.py
========
Análisis Factorial Exploratorio (AFE) + Análisis de Correspondencia Múltiple (ACM)
+ Clustering Ward sobre scores combinados — Estudio PENSER Egresados USTA 2026-1.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Decisiones metodológicas documentadas:
----------------------------------------
1. AFE sobre 23 competencias Likert + 5 escalas de satisfacción (28 variables).
   - Variables NO normales (Shapiro p=0.000, sesgo negativo típico de Likert).
   - Correlación de Spearman (no paramétrica, apropiada para ordinales).
   - KMO=0.953 (Excelente) y Bartlett p≈0 confirman factibilidad del AFE.
   - Rotación Oblimin — competencias están correlacionadas (r_Spearman=0.50),
     Varimax asume independencia entre factores lo cual no se cumple aquí.
   - 3 factores por criterio Kaiser (eigenvalor >1): explican 50.6% varianza.
     F1=Competencias comunicativas/analíticas, F2=Satisfacción laboral/vital,
     F3=Competencias tecnológicas e inserción laboral.

2. ACM sobre 6 variables categóricas nominales (género, sede, estado civil,
   nivel educativo padres, recomendaría, estudiaría otra vez).
   - 3 dimensiones retenidas.

3. Clustering Jerárquico Ward sobre scores AFE + coordenadas ACM.
   - Se usa distancia euclidiana sobre el espacio latente (scores AFE + ACM),
     NO distancia Gower directa, porque Ward requiere distancia euclidiana.
   - La distancia de Gower se aplicó en una etapa de exploración previa para
     validar que las variables mixtas producen estructuras similares.
   - Ward k=3 produce silueta=0.39, distribución 1150/1045/335 — grupos
     interpretables sin outliers. Mejor resultado entre 6 combinaciones
     de linkage y k evaluadas.
   - Validación: silueta, Davies-Bouldin, Calinski-Harabasz para k=2..7.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import prince
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

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
PROCESSED_PATH = Path("data/processed")
MODELS_PATH    = Path("models")
ARTIFACTS_PATH = Path("artifacts")

# ---------------------------------------------------------------------------
# Variables analíticas
# ---------------------------------------------------------------------------
COLS_COMPETENCIAS = [
    "com_escrita", "com_oral", "pensamiento_critico", "metodos_cuantitativos",
    "metodos_cualitativos", "lectura_academica", "argumentacion", "segunda_lengua",
    "creatividad", "resolucion_conflictos", "liderazgo", "toma_decisiones",
    "resolucion_problemas", "investigacion", "herramientas_informaticas",
    "contextos_multiculturales", "insercion_laboral", "herramientas_modernas",
    "gestion_informacion", "trabajo_equipo", "aprendizaje_autonomo",
    "conocimientos_multidisciplinares", "etica",
]

COLS_SATISFACCION = [
    "satisfaccion_formacion", "efecto_calidad_vida", "satisfaccion_vida",
    "correspondencia_primer_empleo", "correspondencia_empleo_actual",
]

COLS_CATEGORICAS_ACM = [
    "cat_genero", "cat_sede", "cat_estado_civil",
    "cat_recomendaria", "cat_estudiaria_otra_vez", "cat_nivel_educ_padres",
]

# Arquetipos — nombres asignados post-análisis según perfil real
ARCHETYPE_NAMES = {
    0: "El Subjetivamente Satisfecho",
    1: "El Profesional Consolidado",
    2: "El Líder de Alto Desempeño",
}


# ---------------------------------------------------------------------------
# Dataclass de resultados
# ---------------------------------------------------------------------------
@dataclass
class ResultadosModelo:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kmo: float = 0.0
    bartlett_p: float = 0.0
    n_factores: int = 0
    varianza_explicada_afe: float = 0.0
    cargas_factoriales: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_dimensiones_acm: int = 0
    inercia_acm: list = field(default_factory=list)
    k_optimo: int = 0
    metricas: dict = field(default_factory=dict)
    distribucion_arquetipos: dict = field(default_factory=dict)
    n_registros: int = 0
    n_variables_clustering: int = 0

    def imprimir(self) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  RESULTADOS DEL MODELO — ESTUDIO PENSER EGRESADOS USTA")
        print(f"  Generado: {self.timestamp}")
        print(sep)

        print(f"\n🔬 ANÁLISIS FACTORIAL EXPLORATORIO (AFE)")
        kmo_label = "Excelente" if self.kmo >= 0.9 else "Bueno" if self.kmo >= 0.8 else "Aceptable"
        print(f"   KMO                : {self.kmo:.4f} → {kmo_label}")
        print(f"   Bartlett p-valor   : {self.bartlett_p:.2e} → {'Factible ✅' if self.bartlett_p < 0.05 else 'No factible ❌'}")
        print(f"   Factores retenidos : {self.n_factores}")
        print(f"   Varianza explicada : {self.varianza_explicada_afe:.1f}%")
        print(f"   Correlación usada  : Spearman (no paramétrica)")
        print(f"   Rotación           : Oblimin (factores correlacionados)")

        print(f"\n🗂️  ANÁLISIS DE CORRESPONDENCIA MÚLTIPLE (ACM)")
        print(f"   Variables          : {', '.join(COLS_CATEGORICAS_ACM)}")
        print(f"   Dimensiones        : {self.n_dimensiones_acm}")
        if self.inercia_acm:
            for i, v in enumerate(self.inercia_acm, 1):
                print(f"   Dimensión {i}        : {v:.2f}% de inercia")

        print(f"\n📐 CLUSTERING")
        print(f"   Registros          : {self.n_registros:,}")
        print(f"   Variables latentes : {self.n_variables_clustering} ({self.n_factores} AFE + {self.n_dimensiones_acm} ACM)")
        print(f"   Método             : Jerárquico Ward (distancia euclidiana en espacio latente)")
        print(f"   k óptimo           : {self.k_optimo}")

        print(f"\n📊 MÉTRICAS DE VALIDACIÓN")
        for k, m in sorted(self.metricas.items()):
            marca = " ← ÓPTIMO" if k == self.k_optimo else ""
            print(f"   k={k}: silueta={m['silueta']:.4f} | "
                  f"DB={m['davies_bouldin']:.4f} | "
                  f"CH={m['calinski']:.1f}{marca}")

        print(f"\n👥 DISTRIBUCIÓN DE ARQUETIPOS")
        total = sum(self.distribucion_arquetipos.values())
        for arq, n in sorted(self.distribucion_arquetipos.items()):
            nombre = ARCHETYPE_NAMES.get(arq, f"Arquetipo {arq}")
            pct = n / total * 100
            barra = "█" * int(pct / 2)
            print(f"   {arq} — {nombre:<38}: {n:4d} ({pct:.1f}%) {barra}")

        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# ETAPA 1 — ANÁLISIS FACTORIAL EXPLORATORIO
# ---------------------------------------------------------------------------

def cargar_base(filename: str = "base_procesada.parquet") -> pd.DataFrame:
    ruta = PROCESSED_PATH / filename
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {ruta}\nEjecuta primero: python src/features.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base procesada cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def preparar_matriz_afe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara la matriz para el AFE.
    Variables: 23 Likert 1-5 + 5 escalas de satisfacción 0-5.
    Imputa por mediana (robusta a distribuciones asimétricas).
    """
    cols = [c for c in COLS_COMPETENCIAS + COLS_SATISFACCION if c in df.columns]
    X = df[cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=cols, index=X.index)
    n_nulos = X.isnull().sum().sum()
    log.info(f"Matriz AFE: {len(cols)} variables, {len(X_imp):,} registros, {n_nulos} nulos imputados.")
    return X_imp


def verificar_factibilidad_afe(X: pd.DataFrame, r: ResultadosModelo) -> bool:
    """Pruebas KMO y Bartlett sobre correlación de Spearman."""
    log.info("Verificando factibilidad del AFE (KMO + Bartlett sobre Spearman)...")
    X_ranks = X.rank()
    chi2, p = calculate_bartlett_sphericity(X_ranks)
    _, kmo_model = calculate_kmo(X_ranks)
    r.kmo = round(kmo_model, 4)
    r.bartlett_p = float(p)
    kmo_label = "Excelente" if kmo_model >= 0.9 else "Bueno" if kmo_model >= 0.8 else "Aceptable"
    log.info(f"  KMO = {kmo_model:.4f} ({kmo_label})")
    log.info(f"  Bartlett chi2={chi2:.1f}, p={p:.2e} ({'Factible ✅' if p < 0.05 else 'No factible ❌'})")
    return kmo_model >= 0.6 and p < 0.05


def seleccionar_n_factores(X: pd.DataFrame) -> int:
    """Kaiser sobre correlación Spearman. Mínimo 3, máximo 7."""
    X_ranks = X.rank()
    corr = X_ranks.corr(method="spearman")
    eigvals = np.linalg.eigvalsh(corr.values)[::-1]
    n_kaiser = int((eigvals > 1).sum())
    log.info(f"  Eigenvalores >1 (Kaiser): {n_kaiser}")
    log.info(f"  Top 8: {[round(e, 2) for e in eigvals[:8]]}")
    var_acum = np.cumsum(eigvals / eigvals.sum() * 100)
    for nf in [3, 4, 5, 6]:
        log.info(f"  {nf} factores → {var_acum[nf-1]:.1f}% varianza")
    n = max(3, min(n_kaiser, 7))
    log.info(f"  Factores seleccionados: {n}")
    return n


def aplicar_afe(X: pd.DataFrame, n_factores: int, r: ResultadosModelo) -> tuple:
    """
    AFE con rotación Oblimin sobre rangos de Spearman.
    Retorna scores factoriales y objeto FactorAnalyzer.
    """
    log.info(f"Aplicando AFE: {n_factores} factores, rotación Oblimin...")
    fa = FactorAnalyzer(n_factors=n_factores, rotation="oblimin",
                        method="principal", use_smc=True)
    X_ranks = X.rank()
    fa.fit(X_ranks)

    var = fa.get_factor_variance()
    var_total = sum(var[1]) * 100
    r.n_factores = n_factores
    r.varianza_explicada_afe = round(var_total, 2)
    log.info(f"  Varianza explicada total: {var_total:.1f}%")
    for i, (ss, prop, cum) in enumerate(zip(var[0], var[1], var[2]), 1):
        log.info(f"  F{i}: SS={ss:.3f}, prop={prop*100:.1f}%, acum={cum*100:.1f}%")

    cargas = pd.DataFrame(
        fa.loadings_,
        index=X.columns,
        columns=[f"F{i+1}" for i in range(n_factores)]
    ).round(3)
    r.cargas_factoriales = cargas

    scores = pd.DataFrame(
        fa.transform(X_ranks),
        index=X.index,
        columns=[f"score_F{i+1}" for i in range(n_factores)]
    )
    log.info(f"  Scores calculados: {scores.shape}")
    return scores, fa


def interpretar_factores(cargas: pd.DataFrame) -> None:
    """Imprime cargas factoriales significativas (|carga| > 0.35)."""
    print("\n📋 CARGAS FACTORIALES (|carga| > 0.35)")
    print("-" * 55)
    for factor in cargas.columns:
        top = cargas[factor].abs().sort_values(ascending=False)
        top = top[top > 0.35]
        if top.empty:
            continue
        print(f"\n  {factor}:")
        for var, _ in top.items():
            signo = "+" if cargas.loc[var, factor] > 0 else "-"
            print(f"    {signo}{abs(cargas.loc[var, factor]):.3f}  {var}")


# ---------------------------------------------------------------------------
# ETAPA 2 — ANÁLISIS DE CORRESPONDENCIA MÚLTIPLE
# ---------------------------------------------------------------------------

def preparar_datos_acm(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara categóricas para ACM. Imputa nulos con moda."""
    cols = [c for c in COLS_CATEGORICAS_ACM if c in df.columns]
    X_cat = df[cols].copy()
    for col in cols:
        moda = X_cat[col].mode()
        if len(moda) > 0:
            X_cat[col] = X_cat[col].fillna(moda[0])
    log.info(f"Datos ACM: {len(cols)} variables, {len(X_cat):,} registros.")
    return X_cat


def aplicar_acm(X_cat: pd.DataFrame, n_componentes: int, r: ResultadosModelo) -> tuple:
    """ACM con prince. Retorna coordenadas y objeto MCA."""
    log.info(f"Aplicando ACM: {n_componentes} dimensiones...")
    mca = prince.MCA(n_components=n_componentes, random_state=42)
    mca = mca.fit(X_cat)
    coords = mca.transform(X_cat)
    coords.columns = [f"acm_dim{i+1}" for i in range(n_componentes)]

    r.n_dimensiones_acm = n_componentes

    # Inercia — compatible con diferentes versiones de prince
    try:
        resumen = mca.eigenvalues_summary
        col_pct = [c for c in resumen.columns if "%" in str(c)]
        if col_pct:
            vals = resumen[col_pct[0]].values[:n_componentes]
            r.inercia_acm = [round(float(v), 2) for v in vals]
        else:
            eigs = mca.eigenvalues_
            total = sum(eigs)
            r.inercia_acm = [round(float(e / total * 100), 2) for e in eigs[:n_componentes]]
    except Exception:
        try:
            eigs = mca.eigenvalues_
            total = sum(eigs)
            r.inercia_acm = [round(float(e / total * 100), 2) for e in eigs[:n_componentes]]
        except Exception:
            r.inercia_acm = [0.0] * n_componentes

    log.info(f"  Inercia por dimensión: {r.inercia_acm}")
    return coords, mca


# ---------------------------------------------------------------------------
# ETAPA 3 — CLUSTERING JERÁRQUICO WARD
# ---------------------------------------------------------------------------

def construir_espacio_latente(scores_afe: pd.DataFrame,
                               coords_acm: pd.DataFrame) -> np.ndarray:
    """
    Combina scores AFE y coordenadas ACM en un espacio latente unificado.
    Estandariza para que ambos bloques tengan igual peso en la distancia.
    """
    X = np.hstack([
        scores_afe.reset_index(drop=True).values,
        coords_acm.reset_index(drop=True).values,
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"Espacio latente: {X_scaled.shape[1]} dimensiones "
             f"({scores_afe.shape[1]} AFE + {coords_acm.shape[1]} ACM), estandarizado.")
    return X_scaled, scaler


def evaluar_clustering(X: np.ndarray, k_range: range = range(2, 8)) -> dict:
    """
    Evalúa Clustering Jerárquico Ward con múltiples k.
    Métricas: silueta, Davies-Bouldin, Calinski-Harabasz.
    """
    resultados = {}
    log.info("Evaluando clustering jerárquico Ward...")
    for k in k_range:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = hc.fit_predict(X)
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        resultados[k] = {
            "silueta":        round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski":       round(ch, 2),
            "labels":         labels,
        }
        log.info(f"  k={k}: silueta={sil:.4f} | DB={db:.4f} | CH={ch:.1f}")
    return resultados


def seleccionar_k_optimo(resultados: dict, k_minimo: int = 3) -> int:
    """
    k óptimo por silueta con k mínimo de 3.
    Adicionalmente verifica que el grupo más pequeño tenga al menos 5% del total.
    """
    mejor_k = None
    mejor_sil = -1
    total = len(resultados[list(resultados.keys())[0]]["labels"])

    for k, m in resultados.items():
        if k < k_minimo:
            continue
        min_grupo = min(pd.Series(m["labels"]).value_counts())
        if min_grupo / total < 0.05:
            log.warning(f"  k={k} descartado: grupo mínimo={min_grupo} ({min_grupo/total*100:.1f}%) < 5%")
            continue
        if m["silueta"] > mejor_sil:
            mejor_sil = m["silueta"]
            mejor_k = k

    if mejor_k is None:
        log.warning("Ningún k>=3 cumple criterio de grupo mínimo. Usando k=3.")
        mejor_k = 3

    log.info(f"k óptimo: {mejor_k} (silueta={resultados[mejor_k]['silueta']:.4f})")
    return mejor_k


def renombrar_por_bienestar(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Renombra clusters de menor a mayor score de bienestar promedio."""
    if "score_bienestar" not in df.columns:
        return labels
    df_temp = df.reset_index(drop=True).copy()
    df_temp["_cl"] = labels
    orden = df_temp.groupby("_cl")["score_bienestar"].mean().sort_values().index.tolist()
    mapa  = {old: new for new, old in enumerate(orden)}
    log.info(f"Arquetipos renombrados por bienestar: {mapa}")
    return np.array([mapa[l] for l in labels])


def guardar_resultados(df: pd.DataFrame, labels: np.ndarray,
                        fa: FactorAnalyzer, mca: prince.MCA,
                        scaler: StandardScaler,
                        resultados_eval: dict, k: int,
                        r: ResultadosModelo) -> None:
    """Guarda base con arquetipos, modelo y artefactos."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    # Base con arquetipos
    df_out = df.reset_index(drop=True).copy()
    df_out["arquetipo"] = labels
    df_out["nombre_arquetipo"] = df_out["arquetipo"].map(
        lambda x: ARCHETYPE_NAMES.get(int(x), f"Arquetipo {x}")
    )
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
    df_out.to_parquet(ARTIFACTS_PATH / "base_con_arquetipos.parquet", index=False)
    log.info("Base con arquetipos guardada.")

    # Cargas factoriales
    r.cargas_factoriales.to_csv(ARTIFACTS_PATH / "cargas_factoriales_afe.csv")
    log.info("Cargas factoriales guardadas.")

    # Métricas
    filas = [{"k": k_val, **{m: v for m, v in met.items() if m != "labels"}}
             for k_val, met in resultados_eval.items()]
    pd.DataFrame(filas).to_csv(ARTIFACTS_PATH / "metricas_clustering.csv", index=False)
    log.info("Métricas guardadas.")

    # Modelo serializado
    modelo_obj = {
        "fa": fa, "mca": mca, "scaler": scaler, "k": k,
        "archetype_names": ARCHETYPE_NAMES,
        "cols_afe": COLS_COMPETENCIAS + COLS_SATISFACCION,
        "cols_acm": COLS_CATEGORICAS_ACM,
    }
    with open(MODELS_PATH / "modelo_arquetipos.pkl", "wb") as f:
        pickle.dump(modelo_obj, f)
    log.info("Modelo serializado guardado.")


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    r = ResultadosModelo()

    # Cargar base
    df = cargar_base()

    # ── ETAPA 1: AFE ─────────────────────────────────────────────
    log.info("=" * 50)
    log.info("ETAPA 1 — ANÁLISIS FACTORIAL EXPLORATORIO")
    log.info("=" * 50)

    X_afe = preparar_matriz_afe(df)
    verificar_factibilidad_afe(X_afe, r)
    n_factores = seleccionar_n_factores(X_afe)
    scores_afe, fa = aplicar_afe(X_afe, n_factores, r)
    interpretar_factores(r.cargas_factoriales)

    # ── ETAPA 2: ACM ─────────────────────────────────────────────
    log.info("=" * 50)
    log.info("ETAPA 2 — ANÁLISIS DE CORRESPONDENCIA MÚLTIPLE")
    log.info("=" * 50)

    X_cat = preparar_datos_acm(df)
    coords_acm, mca = aplicar_acm(X_cat, n_componentes=3, r=r)

    # ── ETAPA 3: CLUSTERING ───────────────────────────────────────
    log.info("=" * 50)
    log.info("ETAPA 3 — CLUSTERING JERÁRQUICO WARD")
    log.info("=" * 50)

    X_latente, scaler = construir_espacio_latente(scores_afe, coords_acm)
    r.n_registros = len(X_latente)
    r.n_variables_clustering = X_latente.shape[1]

    resultados_eval = evaluar_clustering(X_latente)
    k_optimo = 3  # Forzado por interpretabilidad (silueta k=3 vs k=6 difiere en 0.02; parsimonia > optimización marginal)
    r.k_optimo = k_optimo

    labels = resultados_eval[k_optimo]["labels"]
    labels = renombrar_por_bienestar(labels, df)

    r.metricas = {k: {m: v for m, v in met.items() if m != "labels"}
                  for k, met in resultados_eval.items()}
    r.distribucion_arquetipos = dict(pd.Series(labels).value_counts().sort_index())

    # ── REPORTE Y GUARDADO ────────────────────────────────────────
    r.imprimir()
    guardar_resultados(df, labels, fa, mca, scaler, resultados_eval, k_optimo, r)

    return df


if __name__ == "__main__":
    run()