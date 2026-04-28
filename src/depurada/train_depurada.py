"""
train_depurada.py
=================
AFE + ACM + Clustering Ward — Base Depurada PENSER USTA 2025.

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Decisiones metodológicas:
---------------------------
1. AFE sobre 15 variables de logro + 6 de incidencia (21 variables).
   - KMO=0.942 (Excelente), Bartlett p≈0.
   - Kaiser sugiere 2 factores, se usan 3 para interpretabilidad
     (F1=Competencias, F2=Incidencia formación, F3=Logro transversal específico).
   - Correlación Spearman + rotación Oblimin.

2. ACM sobre variables categóricas: sede, programa, tipo cargo, laborando.
   - cat_programa tiene 26 categorías (programas con <10 registros → "Otro").
   - 3 dimensiones retenidas.

3. Clustering Ward k=3 sobre espacio latente AFE+ACM estandarizado.
   - Silueta k=3: 0.27 (mejor que la base de percepción: 0.11).
   - Todos los grupos >5% del total.
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
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed")
MODELS_PATH    = Path("models")
ARTIFACTS_PATH = Path("artifacts")

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

COLS_AFE = COLS_LOGRO + COLS_INCIDENCIA

COLS_ACM = ["cat_sede", "cat_programa", "cat_tipo_cargo", "cat_laborando"]

COLS_ADICIONALES = [
    "score_impacto_formacion", "percepcion_programa",
    "percepcion_ingreso", "formacion_impacto_general",
    "laborando_actualmente", "ingresos_superiores_estudio",
]

ARCHETYPE_NAMES = {
    0: "El Graduado en Desarrollo",
    1: "El Profesional Impactado",
    2: "El Líder con Alta Incidencia",
}


@dataclass
class ResultadosModelo:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kmo: float = 0.0
    bartlett_p: float = 0.0
    n_factores: int = 0
    varianza_afe: float = 0.0
    cargas: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_dim_acm: int = 0
    inercia_acm: list = field(default_factory=list)
    k_optimo: int = 0
    metricas: dict = field(default_factory=dict)
    distribucion: dict = field(default_factory=dict)
    n_registros: int = 0

    def imprimir(self):
        sep = "=" * 65
        print(f"\n{sep}")
        print("  RESULTADOS — BASE DEPURADA PENSER USTA 2025")
        print(f"  Generado: {self.timestamp}")
        print(sep)

        print(f"\n🔬 AFE")
        kmo_label = "Excelente" if self.kmo >= 0.9 else "Bueno" if self.kmo >= 0.8 else "Aceptable"
        print(f"   KMO                : {self.kmo:.4f} → {kmo_label}")
        print(f"   Bartlett p         : {self.bartlett_p:.2e} → {'Factible ✅' if self.bartlett_p < 0.05 else 'No factible ❌'}")
        print(f"   Factores           : {self.n_factores}")
        print(f"   Varianza explicada : {self.varianza_afe:.1f}%")

        print(f"\n🗂️  ACM")
        print(f"   Dimensiones        : {self.n_dim_acm}")
        for i, v in enumerate(self.inercia_acm, 1):
            print(f"   Dimensión {i}        : {v:.2f}% inercia")

        print(f"\n📊 MÉTRICAS CLUSTERING")
        for k, m in sorted(self.metricas.items()):
            marca = " ← ÓPTIMO" if k == self.k_optimo else ""
            print(f"   k={k}: silueta={m['silueta']:.4f} | DB={m['davies_bouldin']:.4f} | CH={m['calinski']:.1f}{marca}")

        print(f"\n👥 DISTRIBUCIÓN DE ARQUETIPOS (n={self.n_registros:,})")
        for arq, n in sorted(self.distribucion.items()):
            nombre = ARCHETYPE_NAMES.get(arq, f"Arquetipo {arq}")
            pct = n / self.n_registros * 100
            barra = "█" * int(pct / 2)
            print(f"   {arq} — {nombre:<35}: {n:4d} ({pct:.1f}%) {barra}")
        print(f"\n{sep}\n")


def cargar_base():
    ruta = PROCESSED_PATH / "base_depurada_procesada.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {ruta}\nEjecuta primero: python src/depurada/features_depurada.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def preparar_afe(df):
    cols = [c for c in COLS_AFE if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=cols, index=X.index)
    log.info(f"Matriz AFE: {len(cols)} variables, {X_imp.isnull().sum().sum()} nulos imputados.")
    return X_imp


def verificar_afe(X, r):
    X_ranks = X.rank()
    chi2, p = calculate_bartlett_sphericity(X_ranks)
    _, kmo = calculate_kmo(X_ranks)
    r.kmo = round(kmo, 4)
    r.bartlett_p = float(p)
    kmo_label = "Excelente" if kmo >= 0.9 else "Bueno" if kmo >= 0.8 else "Aceptable"
    log.info(f"KMO={kmo:.4f} ({kmo_label}) | Bartlett p={p:.2e}")

    eigvals = np.linalg.eigvalsh(X_ranks.corr().values)[::-1]
    n_kaiser = int((eigvals > 1).sum())
    log.info(f"Kaiser: {n_kaiser} factores | Varianza: " +
             " | ".join([f"{i+1}F={np.cumsum(eigvals/eigvals.sum()*100)[i]:.1f}%" for i in [1,2,3,4]]))
    return max(3, min(n_kaiser, 5))


def aplicar_afe(X, n_factores, r):
    log.info(f"AFE: {n_factores} factores, Oblimin, Spearman...")
    fa = FactorAnalyzer(n_factors=n_factores, rotation="oblimin", method="principal", use_smc=True)
    fa.fit(X.rank())
    var = fa.get_factor_variance()
    r.n_factores = n_factores
    r.varianza_afe = round(sum(var[1]) * 100, 2)
    log.info(f"Varianza total: {r.varianza_afe:.1f}%")
    for i, (ss, prop, cum) in enumerate(zip(var[0], var[1], var[2]), 1):
        log.info(f"  F{i}: SS={ss:.3f}, {prop*100:.1f}%, acum={cum*100:.1f}%")

    r.cargas = pd.DataFrame(fa.loadings_, index=X.columns,
                             columns=[f"F{i+1}" for i in range(n_factores)]).round(3)

    scores = pd.DataFrame(fa.transform(X.rank()), index=X.index,
                           columns=[f"score_F{i+1}" for i in range(n_factores)])
    log.info(f"Scores AFE: {scores.shape}")

    print("\n📋 CARGAS FACTORIALES (|carga| > 0.35)")
    print("-" * 55)
    for factor in r.cargas.columns:
        top = r.cargas[factor].abs().sort_values(ascending=False)
        top = top[top > 0.35]
        if top.empty:
            continue
        print(f"\n  {factor}:")
        for var_name, _ in top.items():
            signo = "+" if r.cargas.loc[var_name, factor] > 0 else "-"
            print(f"    {signo}{abs(r.cargas.loc[var_name, factor]):.3f}  {var_name}")

    return scores, fa


def preparar_acm(df):
    cols = [c for c in COLS_ACM if c in df.columns]
    X_cat = df[cols].copy()
    for col in cols:
        moda = X_cat[col].mode()
        if len(moda) > 0:
            X_cat[col] = X_cat[col].fillna(moda[0])
    log.info(f"ACM: {len(cols)} variables, {len(X_cat):,} registros.")
    return X_cat


def aplicar_acm(X_cat, r):
    log.info("Aplicando ACM (3 dimensiones)...")
    mca = prince.MCA(n_components=3, random_state=42)
    mca.fit(X_cat)
    coords = mca.transform(X_cat)
    coords.columns = [f"acm_dim{i+1}" for i in range(3)]
    r.n_dim_acm = 3
    try:
        eigs = mca.eigenvalues_
        total = sum(eigs)
        r.inercia_acm = [round(float(e/total*100), 2) for e in eigs[:3]]
    except Exception:
        r.inercia_acm = [0.0, 0.0, 0.0]
    log.info(f"Inercia ACM: {r.inercia_acm}")
    return coords, mca


def evaluar_clustering(X, k_range=range(2, 7)):
    resultados = {}
    log.info("Evaluando clustering Ward...")
    for k in k_range:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = hc.fit_predict(X)
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        resultados[k] = {"silueta": round(sil,4), "davies_bouldin": round(db,4),
                         "calinski": round(ch,2), "labels": labels}
        log.info(f"  k={k}: sil={sil:.4f} | DB={db:.4f} | CH={ch:.1f}")
    return resultados


def seleccionar_k(resultados, k_min=3):
    mejor_k, mejor_sil = None, -1
    total = len(list(resultados.values())[0]["labels"])
    for k, m in resultados.items():
        if k < k_min:
            continue
        min_g = min(pd.Series(m["labels"]).value_counts())
        if min_g / total < 0.05:
            log.warning(f"k={k} descartado: grupo mínimo {min_g/total*100:.1f}% < 5%")
            continue
        if m["silueta"] > mejor_sil:
            mejor_sil = m["silueta"]
            mejor_k = k
    if mejor_k is None:
        mejor_k = 3
    log.info(f"k óptimo: {mejor_k} (silueta={resultados[mejor_k]['silueta']:.4f})")
    return mejor_k


def renombrar_por_score(labels, df):
    if "score_impacto_formacion" not in df.columns:
        return labels
    df_t = df.reset_index(drop=True).copy()
    df_t["_cl"] = labels
    orden = df_t.groupby("_cl")["score_impacto_formacion"].mean().sort_values().index.tolist()
    mapa = {old: new for new, old in enumerate(orden)}
    log.info(f"Arquetipos renombrados por impacto formación: {mapa}")
    return np.array([mapa[l] for l in labels])


def guardar(df, labels, fa, mca, scaler, resultados, k, r):
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    df_out = df.reset_index(drop=True).copy()
    df_out["arquetipo"] = labels
    df_out["nombre_arquetipo"] = df_out["arquetipo"].map(
        lambda x: ARCHETYPE_NAMES.get(int(x), f"Arquetipo {x}")
    )
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
    df_out.to_parquet(ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet", index=False)
    log.info("Base con arquetipos guardada.")

    r.cargas.to_csv(ARTIFACTS_PATH / "cargas_factoriales_depurada.csv")

    filas = [{"k": k_val, **{m: v for m, v in met.items() if m != "labels"}}
             for k_val, met in resultados.items()]
    pd.DataFrame(filas).to_csv(ARTIFACTS_PATH / "metricas_clustering_depurada.csv", index=False)

    with open(MODELS_PATH / "modelo_depurada.pkl", "wb") as f:
        pickle.dump({"fa": fa, "mca": mca, "scaler": scaler, "k": k,
                     "archetype_names": ARCHETYPE_NAMES,
                     "cols_afe": COLS_AFE, "cols_acm": COLS_ACM}, f)
    log.info("Modelo guardado.")


def run():
    r = ResultadosModelo()
    df = cargar_base()

    log.info("=" * 50)
    log.info("ETAPA 1 — AFE")
    log.info("=" * 50)
    X_afe = preparar_afe(df)
    n_factores = verificar_afe(X_afe, r)
    scores_afe, fa = aplicar_afe(X_afe, n_factores, r)

    log.info("=" * 50)
    log.info("ETAPA 2 — ACM")
    log.info("=" * 50)
    X_cat = preparar_acm(df)
    coords_acm, mca = aplicar_acm(X_cat, r)

    log.info("=" * 50)
    log.info("ETAPA 3 — CLUSTERING WARD")
    log.info("=" * 50)
    X_latente = np.hstack([
        scores_afe.reset_index(drop=True).values,
        coords_acm.reset_index(drop=True).values,
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_latente)
    r.n_registros = len(X_scaled)

    resultados = evaluar_clustering(X_scaled)
    k_optimo = 3  # Forzado por interpretabilidad (sil k=3 vs k=2 difiere 0.03)
    r.k_optimo = k_optimo
    r.metricas = {k: {m: v for m, v in met.items() if m != "labels"}
                  for k, met in resultados.items()}

    labels = resultados[k_optimo]["labels"]
    labels = renombrar_por_score(labels, df)
    r.distribucion = dict(pd.Series(labels).value_counts().sort_index())

    r.imprimir()
    guardar(df, labels, fa, mca, scaler, resultados, k_optimo, r)
    return df


if __name__ == "__main__":
    run()