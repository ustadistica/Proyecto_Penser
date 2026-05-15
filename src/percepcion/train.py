"""
train.py — Versión 4.0
======================
AFE por Bloques + ACM + Clustering múltiple con validación robusta
Base: Estudio de Percepción Egresados USTA 2026-1
 
Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1
 
Metodología:
------------
1. AFE por bloques temáticos (5 bloques, 6 indicadores):
   - B1: Competencias Cognitivo-Comunicativas  (KMO=0.924) → 1 factor
   - B2: Competencias Tecnológicas e Inserción (KMO=0.869) → 1 factor
   - B3: Liderazgo y Competencias Sociales     (KMO=0.905) → 1 factor
   - B4a: Satisfacción Vital y Formativa       (KMO=0.673) → 1 factor
   - B4b: Correspondencia Laboral              (KMO=0.673) → 1 factor
   - B5: Bienestar (material + social)         (KMO=0.757) → 2 factores
 
2. ACM sobre variables categóricas nominales → 3 dimensiones
 
3. Estandarización del espacio latente (9 indicadores)
 
4. Clustering con 3 métodos:
   - Ward Jerárquico
   - K-Prototypes (sobre variables originales)
   - DBSCAN (exploración de densidad)
 
5. Validación:
   - Coeficiente de Silueta
   - Índice de Dunn
   - Davies-Bouldin
   - Balance (CV y min% por grupo)
 
6. Selección: k óptimo + segundo mejor k documentados
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
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                              silhouette_score, pairwise_distances)
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
 
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
 
# ---------------------------------------------------------------------------
# BLOQUES AFE
# ---------------------------------------------------------------------------
BLOQUES_AFE = {
    "B1_Cognitivo_Comunicativo": [
        "com_escrita", "com_oral", "pensamiento_critico", "lectura_academica",
        "argumentacion", "creatividad", "metodos_cualitativos", "metodos_cuantitativos"
    ],
    "B2_Tecnologico_Insercion": [
        "herramientas_modernas", "insercion_laboral", "gestion_informacion",
        "herramientas_informaticas", "contextos_multiculturales",
        "conocimientos_multidisciplinares", "aprendizaje_autonomo"
    ],
    "B3_Liderazgo_Social": [
        "liderazgo", "toma_decisiones", "resolucion_problemas",
        "resolucion_conflictos", "trabajo_equipo", "investigacion", "etica"
    ],
    "B4a_Satisfaccion_Vital": [
        "satisfaccion_formacion", "efecto_calidad_vida", "satisfaccion_vida"
    ],
    "B4b_Correspondencia_Laboral": [
        "correspondencia_primer_empleo", "correspondencia_empleo_actual"
    ],
    "B5_Bienestar": [
        "adquirio_bienes", "mejoro_vivienda", "mejoro_salud",
        "acceso_seguridad_social", "incremento_cultural",
        "satisfecho_ocio", "red_amigos"
    ],
}
 
# B5 necesita 2 factores (bienestar material vs social)
N_FACTORES_BLOQUE = {
    "B1_Cognitivo_Comunicativo":  1,
    "B2_Tecnologico_Insercion":   1,
    "B3_Liderazgo_Social":        1,
    "B4a_Satisfaccion_Vital":     1,
    "B4b_Correspondencia_Laboral":1,
    "B5_Bienestar":               2,
}
 
COLS_ACM = [
    "cat_genero", "cat_sede", "cat_estado_civil",
    "cat_recomendaria", "cat_estudiaria_otra_vez", "cat_nivel_educ_padres",
]
 
COLS_CATEGORICAS_KPROTO = [
    "cat_genero", "cat_sede", "cat_estado_civil",
    "cat_recomendaria", "cat_estudiaria_otra_vez",
]
 
ARCHETYPE_NAMES_K3 = {
    0: "El Subjetivamente Satisfecho",
    1: "El Profesional Consolidado",
    2: "El Líder de Alto Desempeño",
}
 
ARCHETYPE_NAMES_K4 = {
    0: "El Graduado en Desarrollo",
    1: "El Profesional Técnico",
    2: "El Profesional Consolidado",
    3: "El Líder de Alto Desempeño",
}
 
 
# ---------------------------------------------------------------------------
# FUNCIONES DE VALIDACIÓN
# ---------------------------------------------------------------------------
 
def dunn_index(X: np.ndarray, labels: np.ndarray, n_sample: int = 1000) -> float:
    """
    Índice de Dunn = min(distancia inter-cluster) / max(diámetro intra-cluster).
    Mayor = mejor separación entre clusters.
    Se submuestra para eficiencia computacional.
    """
    np.random.seed(42)
    idx = np.random.choice(len(X), min(n_sample, len(X)), replace=False)
    X_s, l_s = X[idx], labels[idx]
    D = pairwise_distances(X_s)
    unique = np.unique(l_s)
 
    min_inter = np.inf
    for i in unique:
        for j in unique:
            if i >= j: continue
            d = D[np.ix_(l_s == i, l_s == j)].min()
            min_inter = min(min_inter, d)
 
    max_intra = 0
    for i in unique:
        mask = l_s == i
        if mask.sum() < 2: continue
        d = D[np.ix_(mask, mask)].max()
        max_intra = max(max_intra, d)
 
    return float(min_inter / max_intra) if max_intra > 0 else 0.0
 
 
def balance_score(labels: np.ndarray) -> dict:
    """
    Métricas de balance de un clustering.
    CV bajo = bien balanceado.
    min_pct alto = ningún grupo demasiado pequeño.
    """
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    return {
        "cv":      round(float(counts.std() / counts.mean()), 4),
        "min_pct": round(float(counts.min() / total * 100), 2),
        "max_pct": round(float(counts.max() / total * 100), 2),
        "dist":    dict(counts),
    }
 
 
def score_compuesto(sil, db, dunn, cv) -> float:
    """
    Score compuesto normalizado para comparar configuraciones.
    Pondera: silueta (35%) + Dunn (25%) + DB inverso (25%) + balance (15%)
    """
    return sil * 0.35 + dunn * 0.25 + (1 / (1 + db)) * 0.25 + (1 / (1 + cv)) * 0.15
 
 
# ---------------------------------------------------------------------------
# ETAPA 1: AFE POR BLOQUES
# ---------------------------------------------------------------------------
 
def cargar_base():
    ruta = PROCESSED_PATH / "base_procesada.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No encontrado: {ruta}\nEjecuta: python src/percepcion/features.py")
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} × {df.shape[1]}")
    return df
 
 
def afe_por_bloques(df: pd.DataFrame) -> tuple:
    """
    Aplica AFE independiente a cada bloque temático.
    Retorna: scores (DataFrame), resumen de factibilidad por bloque.
    """
    imp = SimpleImputer(strategy="median")
    scores_bloques = {}
    resumen_bloques = {}
 
    log.info("=" * 55)
    log.info("ETAPA 1 — AFE POR BLOQUES")
    log.info("=" * 55)
 
    for nombre, cols in BLOQUES_AFE.items():
        cols_ok = [c for c in cols if c in df.columns]
        if not cols_ok:
            log.warning(f"  {nombre}: sin columnas disponibles")
            continue
 
        X = df[cols_ok].apply(pd.to_numeric, errors="coerce")
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=cols_ok, index=df.index)
        X_ranks = X_imp.rank()
 
        # KMO y Bartlett
        try:
            chi2, p = calculate_bartlett_sphericity(X_ranks)
            _, kmo = calculate_kmo(X_ranks)
        except Exception:
            kmo, p = 0.0, 1.0
 
        kmo_label = ("Excelente" if kmo >= 0.9 else "Bueno" if kmo >= 0.8
                     else "Aceptable" if kmo >= 0.7 else "Mediocre")
 
        n_factores = N_FACTORES_BLOQUE.get(nombre, 1)
 
        # AFE
        try:
            fa = FactorAnalyzer(
                n_factors=n_factores, rotation="oblimin" if n_factores > 1 else "varimax",
                method="principal", use_smc=True
            )
            fa.fit(X_ranks)
            var = fa.get_factor_variance()
            var_total = sum(var[1]) * 100
 
            scores = pd.DataFrame(
                fa.transform(X_ranks),
                index=df.index,
                columns=[f"{nombre}_F{i+1}" for i in range(n_factores)]
            )
            for col in scores.columns:
                scores_bloques[col] = scores[col]
 
            cargas = pd.DataFrame(
                fa.loadings_, index=cols_ok,
                columns=[f"F{i+1}" for i in range(n_factores)]
            ).round(3)
 
            resumen_bloques[nombre] = {
                "kmo": round(kmo, 4), "kmo_label": kmo_label,
                "bartlett_p": float(p), "n_factores": n_factores,
                "varianza_total": round(var_total, 2),
                "cargas": cargas, "n_vars": len(cols_ok),
            }
 
            log.info(f"  {nombre}: KMO={kmo:.4f} ({kmo_label}) | "
                     f"{n_factores}F | var={var_total:.1f}%")
 
        except Exception as e:
            log.warning(f"  {nombre}: ERROR AFE — {e}")
            continue
 
    scores_df = pd.DataFrame(scores_bloques, index=df.index)
    log.info(f"  → {scores_df.shape[1]} indicadores AFE generados")
    return scores_df, resumen_bloques
 
 
def imprimir_resumen_afe(resumen: dict) -> None:
    print("\n" + "=" * 65)
    print("  AFE POR BLOQUES — RESUMEN DE FACTIBILIDAD")
    print("=" * 65)
    for nombre, info in resumen.items():
        print(f"\n  {nombre}")
        print(f"    KMO      : {info['kmo']:.4f} → {info['kmo_label']}")
        print(f"    Bartlett : p={info['bartlett_p']:.2e} "
              f"{'✅' if info['bartlett_p'] < 0.05 else '❌'}")
        print(f"    Factores : {info['n_factores']}")
        print(f"    Varianza : {info['varianza_total']:.1f}%")
        print(f"    Variables: {info['n_vars']}")
        if "cargas" in info:
            print(f"    Cargas (|>0.5|):")
            for factor in info["cargas"].columns:
                top = info["cargas"][factor].abs()
                top = top[top > 0.5].sort_values(ascending=False)
                if not top.empty:
                    vars_str = ", ".join([f"{v}({info['cargas'].loc[v,factor]:+.2f})"
                                          for v in top.index[:3]])
                    print(f"      {factor}: {vars_str}")
 
 
# ---------------------------------------------------------------------------
# ETAPA 2: ACM
# ---------------------------------------------------------------------------
 
def aplicar_acm(df: pd.DataFrame) -> tuple:
    """ACM sobre variables categóricas nominales."""
    log.info("=" * 55)
    log.info("ETAPA 2 — ACM")
    log.info("=" * 55)
 
    cols = [c for c in COLS_ACM if c in df.columns]
    X_cat = df[cols].copy()
    for col in cols:
        moda = X_cat[col].mode()
        X_cat[col] = X_cat[col].fillna(moda[0] if len(moda) > 0 else "Otro")
 
    mca = prince.MCA(n_components=3, random_state=42)
    mca.fit(X_cat)
    coords = mca.transform(X_cat)
    coords.columns = [f"acm_dim{i+1}" for i in range(3)]
    coords.index = df.index
 
    try:
        eigs = mca.eigenvalues_
        total = sum(eigs)
        inercia = [round(float(e/total*100), 2) for e in eigs[:3]]
    except Exception:
        inercia = [0.0, 0.0, 0.0]
 
    log.info(f"  ACM: {len(cols)} vars | inercia: {inercia}")
    return coords, mca, inercia
 
 
# ---------------------------------------------------------------------------
# ETAPA 3: ESPACIO LATENTE
# ---------------------------------------------------------------------------
 
def construir_espacio_latente(scores_afe: pd.DataFrame,
                               coords_acm: pd.DataFrame,
                               n_components: int = 3) -> tuple:
    """
    Combina indicadores AFE + coordenadas ACM, estandariza y aplica PCA.
 
    El PCA sobre el espacio latente AFE+ACM mejora la separación entre
    clusters al maximizar la varianza explicada y reducir el ruido residual
    de los scores factoriales. Con datos Likert ordinales esta reducción
    es especialmente beneficiosa para la silueta del clustering.
 
    n_components=3 retiene ~63% de varianza en percepción y ~70% en depurada.
    """
    from sklearn.decomposition import PCA
 
    X = pd.concat([scores_afe, coords_acm], axis=1)
 
    # QuantileTransformer(normal) para datos ordinales Likert
    # Normaliza la distribución antes del PCA — mejora la separación de clusters
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_qt = qt.fit_transform(X)
 
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_qt)
    var_exp = pca.explained_variance_ratio_.sum() * 100
 
    log.info(f"  Espacio latente: {X.shape[1]} dims "
             f"({scores_afe.shape[1]} AFE + {coords_acm.shape[1]} ACM)")
    log.info(f"  QuantileTransformer(normal) aplicado")
    log.info(f"  PCA: {n_components} componentes → {var_exp:.1f}% varianza explicada")
    log.info(f"  Por componente: {[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
 
    return X_pca, qt, pca, X.columns.tolist()
 
 
# ---------------------------------------------------------------------------
# ETAPA 4: CLUSTERING
# ---------------------------------------------------------------------------
 
def evaluar_ward(X: np.ndarray, k_range=range(2, 8)) -> dict:
    """Ward jerárquico sobre espacio latente."""
    resultados = {}
    log.info("Ward jerárquico...")
    for k in k_range:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = hc.fit_predict(X)
        sil  = silhouette_score(X, labels)
        db   = davies_bouldin_score(X, labels)
        dunn = dunn_index(X, labels)
        bal  = balance_score(labels)
        sc   = score_compuesto(sil, db, dunn, bal["cv"])
        resultados[k] = {
            "labels": labels, "sil": round(sil, 4), "db": round(db, 4),
            "dunn": round(dunn, 4), "bal": bal, "score": round(sc, 4)
        }
        log.info(f"  k={k}: sil={sil:.4f} | Dunn={dunn:.4f} | DB={db:.4f} | "
                 f"CV={bal['cv']:.3f} | min={bal['min_pct']:.1f}% | score={sc:.4f}")
    return resultados
 
 
 
 
def evaluar_kmeans(X: np.ndarray, k_range=range(2, 8)) -> dict:
    """
    KMeans sobre espacio latente QuantileNormal+PCA.
    Método principal recomendado para datos Likert ordinales.
    KMeans minimiza la varianza intra-cluster y produce mejores
    siluetas que Ward con datos de distribución unimodal.
    """
    resultados = {}
    log.info("KMeans (método principal)...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500)
        labels = km.fit_predict(X)
        sil   = silhouette_score(X, labels)
        db    = davies_bouldin_score(X, labels)
        dunn  = dunn_index(X, labels)
        bal   = balance_score(labels)
        sc    = score_compuesto(sil, db, dunn, bal["cv"])
        resultados[k] = {
            "labels": labels, "sil": round(sil, 4), "db": round(db, 4),
            "dunn": round(dunn, 4), "bal": bal, "score": round(sc, 4)
        }
        log.info(f"  k={k}: sil={sil:.4f} | Dunn={dunn:.4f} | DB={db:.4f} | "
                 f"CV={bal['cv']:.3f} | min={bal['min_pct']:.1f}% | score={sc:.4f}")
    return resultados
 
def evaluar_kprototypes(df: pd.DataFrame, X_latente: np.ndarray,
                         k_range=range(2, 8)) -> dict:
    """
    K-Prototypes sobre variables ORIGINALES (su dominio natural).
    La silueta se calcula sobre el espacio latente para comparación justa.
    """
    resultados = {}
    log.info("K-Prototypes (variables originales)...")
 
    # Variables numéricas: todas las Likert + satisfacción
    num_cols = [c for c in (list(BLOQUES_AFE["B1_Cognitivo_Comunicativo"]) +
                             list(BLOQUES_AFE["B2_Tecnologico_Insercion"]) +
                             list(BLOQUES_AFE["B3_Liderazgo_Social"]) +
                             list(BLOQUES_AFE["B4a_Satisfaccion_Vital"]) +
                             list(BLOQUES_AFE["B4b_Correspondencia_Laboral"])) if c in df.columns]
 
    cat_cols_kp = [c for c in COLS_CATEGORICAS_KPROTO if c in df.columns]
 
    imp = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(imp.fit_transform(
        df[num_cols].apply(pd.to_numeric, errors="coerce")), columns=num_cols)
    X_cat_kp = df[cat_cols_kp].fillna("Otro").astype(str)
 
    X_kp = np.hstack([X_num.values, X_cat_kp.values])
    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols_kp)))
 
    for k in k_range:
        try:
            kp = KPrototypes(n_clusters=k, init="Cao", random_state=42,
                             n_init=3, verbose=0)
            labels = kp.fit_predict(X_kp, categorical=cat_idx)
            sil  = silhouette_score(X_latente, labels)
            db   = davies_bouldin_score(X_latente, labels)
            dunn = dunn_index(X_latente, labels)
            bal  = balance_score(labels)
            sc   = score_compuesto(sil, db, dunn, bal["cv"])
            resultados[k] = {
                "labels": labels, "sil": round(sil, 4), "db": round(db, 4),
                "dunn": round(dunn, 4), "bal": bal, "score": round(sc, 4),
                "modelo": kp
            }
            log.info(f"  k={k}: sil={sil:.4f} | Dunn={dunn:.4f} | DB={db:.4f} | "
                     f"CV={bal['cv']:.3f} | min={bal['min_pct']:.1f}% | score={sc:.4f}")
        except Exception as e:
            log.warning(f"  k={k}: ERROR — {e}")
    return resultados
 
 
def evaluar_dbscan(X: np.ndarray) -> dict:
    """DBSCAN — exploración de estructura de densidad."""
    resultados = {}
    log.info("DBSCAN (exploración)...")
    for eps in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        for min_s in [20, 30, 50]:
            dbs = DBSCAN(eps=eps, min_samples=min_s)
            labels = dbs.fit_predict(X)
            n_cl    = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(labels) * 100
            if n_cl >= 2 and noise_pct < 20:
                mask = labels != -1
                sil  = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else -1
                bal  = balance_score(labels[labels != -1])
                resultados[f"eps{eps}_min{min_s}"] = {
                    "n_clusters": n_cl, "noise_pct": round(noise_pct, 1),
                    "sil": round(sil, 4), "bal": bal
                }
                log.info(f"  eps={eps} min={min_s}: k={n_cl} | "
                         f"ruido={noise_pct:.1f}% | sil={sil:.4f} | "
                         f"min={bal['min_pct']:.1f}%")
    if not resultados:
        log.info("  DBSCAN: ninguna configuración viable (<2 clusters o >20% ruido)")
        log.info("  → Los datos Likert no tienen estructura de densidad clara")
    return resultados
 
 
def seleccionar_mejor_k(resultados: dict, metodo: str,
                         k_min: int = 3) -> tuple:
    """
    Selecciona k óptimo y segundo mejor basado en score compuesto.
 
    - k óptimo: mejor score entre k>=k_min con min%>=15
    - 2do mejor: segundo mejor score entre TODOS los k evaluados
      distintos del óptimo (sin restricción de balance), para
      garantizar que siempre sea diferente al óptimo.
 
    Esta separación de criterios permite comparar el mejor resultado
    balanceado vs el segundo candidato global, enriqueciendo el análisis.
    """
    # k óptimo: balance garantizado
    candidatos_bal = {
        k: v for k, v in resultados.items()
        if k >= k_min and v["bal"]["min_pct"] >= 15
    }
    if not candidatos_bal:
        log.warning(f"  {metodo}: ningún k>={k_min} con min%>=15. Relajando a min%>=10")
        candidatos_bal = {k: v for k, v in resultados.items()
                          if k >= k_min and v["bal"]["min_pct"] >= 10}
    if not candidatos_bal:
        candidatos_bal = {k: v for k, v in resultados.items() if k >= k_min}
    if not candidatos_bal:
        candidatos_bal = resultados
 
    ordenados_bal = sorted(candidatos_bal.items(), key=lambda x: x[1]["score"], reverse=True)
    k_opt = ordenados_bal[0][0]
 
    # 2do mejor: cualquier k distinto al óptimo, ordenado por score global
    candidatos_sec = {k: v for k, v in resultados.items() if k != k_opt}
    if candidatos_sec:
        ordenados_sec = sorted(candidatos_sec.items(), key=lambda x: x[1]["score"], reverse=True)
        k_second = ordenados_sec[0][0]
    else:
        k_second = k_opt
 
    log.info(f"  {metodo} → k óptimo={k_opt} "
             f"(score={resultados[k_opt]['score']:.4f}, min%={resultados[k_opt]['bal']['min_pct']:.1f}%) | "
             f"2do mejor={k_second} "
             f"(score={resultados[k_second]['score']:.4f}, min%={resultados[k_second]['bal']['min_pct']:.1f}%)")
    return k_opt, k_second
 
 
# ---------------------------------------------------------------------------
# ETAPA 5: REPORTE Y GUARDADO
# ---------------------------------------------------------------------------
 
def imprimir_comparacion_v2(kmeans_res, ward_res, dbscan_res,
                              k_opt_km, k_sec_km, k_opt_w, k_sec_w):
    """Imprime tabla comparativa KMeans (principal) vs Ward (validación)."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  COMPARACIÓN DE MÉTODOS DE CLUSTERING")
    print(sep)
    print(f"\n{'MÉTODO':<15} {'k':>3} {'Silueta':>9} {'Dunn':>9} {'DB':>9} "
          f"{'CV':>8} {'min%':>7} {'score':>8}")
    print("-" * 65)
    for k, v in sorted(kmeans_res.items()):
        marca = "← ÓPT" if k==k_opt_km else ("← 2do" if k==k_sec_km else "")
        print(f"{'KMeans':<15} {k:>3} {v['sil']:>9.4f} {v['dunn']:>9.4f} "
              f"{v['db']:>9.4f} {v['bal']['cv']:>8.3f} {v['bal']['min_pct']:>7.1f}% "
              f"{v['score']:>8.4f} {marca}")
    print()
    for k, v in sorted(ward_res.items()):
        marca = "← ÓPT" if k==k_opt_w else ("← 2do" if k==k_sec_w else "")
        print(f"{'Ward':<15} {k:>3} {v['sil']:>9.4f} {v['dunn']:>9.4f} "
              f"{v['db']:>9.4f} {v['bal']['cv']:>8.3f} {v['bal']['min_pct']:>7.1f}% "
              f"{v['score']:>8.4f} {marca}")
    if dbscan_res:
        print("\nDBSCAN viables:")
        for cfg, v in dbscan_res.items():
            print(f"  {cfg}: k={v['n_clusters']} | ruido={v['noise_pct']}% | sil={v['sil']:.4f}")
    else:
        print("\nDBSCAN: No viable — datos Likert sin estructura de densidad")
    print(f"\n{sep}")
 
 
def imprimir_comparacion(ward_res: dict, kmeans_res: dict,
                          dbscan_res: dict, k_opt_w: int, k_sec_w: int,
                          k_opt_km: int, k_sec_km: int) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  COMPARACIÓN DE MÉTODOS DE CLUSTERING")
    print(sep)
 
    print(f"\n{'MÉTODO':<15} {'k':>3} {'Silueta':>9} {'Dunn':>9} {'DB':>9} "
          f"{'CV':>8} {'min%':>7} {'score':>8}")
    print("-" * 65)
 
    for k, v in sorted(ward_res.items()):
        marca = "← ÓPT" if k == k_opt_w else ("← 2do" if k == k_sec_w else "")
        print(f"{'Ward':<15} {k:>3} {v['sil']:>9.4f} {v['dunn']:>9.4f} "
              f"{v['db']:>9.4f} {v['bal']['cv']:>8.3f} {v['bal']['min_pct']:>7.1f}% "
              f"{v['score']:>8.4f} {marca}")
 
    print()
    for k, v in sorted(kmeans_res.items()):
        marca = "← ÓPT" if k == k_opt_km else ("← 2do" if k == k_sec_km else "")
        print(f"{'KPrototypes':<15} {k:>3} {v['sil']:>9.4f} {v['dunn']:>9.4f} "
              f"{v['db']:>9.4f} {v['bal']['cv']:>8.3f} {v['bal']['min_pct']:>7.1f}% "
              f"{v['score']:>8.4f} {marca}")
 
    if dbscan_res:
        print()
        print("DBSCAN — configuraciones viables:")
        for cfg, v in dbscan_res.items():
            print(f"  {cfg}: k={v['n_clusters']} | ruido={v['noise_pct']}% | "
                  f"sil={v['sil']:.4f} | min={v['bal']['min_pct']:.1f}%")
    else:
        print("\nDBSCAN: No viable — datos Likert sin estructura de densidad")
 
    print(f"\n{sep}")
 
 
def renombrar_por_bienestar(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Renombra clusters de menor a mayor score de bienestar."""
    if "score_bienestar" not in df.columns:
        return labels
    df_t = df.reset_index(drop=True).copy()
    df_t["_cl"] = labels
    orden = df_t.groupby("_cl")["score_bienestar"].mean().sort_values().index.tolist()
    mapa = {old: new for new, old in enumerate(orden)}
    log.info(f"  Renombrado por bienestar: {mapa}")
    return np.array([mapa[l] for l in labels])
 
 
def guardar_todo(df: pd.DataFrame,
                 labels_km_opt: np.ndarray, labels_km_sec: np.ndarray,
                 labels_ward_opt: np.ndarray, labels_ward_sec: np.ndarray,
                 k_opt_km: int, k_sec_km: int, k_opt_w: int, k_sec_w: int,
                 kmeans_res: dict, ward_res: dict, dbscan_res: dict,
                 resumen_afe: dict, mca: prince.MCA, qt=None, pca_model=None) -> None:
 
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
 
    archetype_names = ARCHETYPE_NAMES_K4 if k_opt_w == 4 else ARCHETYPE_NAMES_K3
 
    # Base con arquetipos (Ward óptimo como principal)
    df_out = df.reset_index(drop=True).copy()
    df_out["arquetipo_kmeans_opt"] = labels_km_opt
    df_out["arquetipo_kmeans_sec"] = labels_km_sec
    df_out["arquetipo_ward_opt"]   = labels_ward_opt
    df_out["arquetipo_ward_sec"]   = labels_ward_sec
    df_out["arquetipo"]            = labels_km_opt
    df_out["nombre_arquetipo"]     = df_out["arquetipo"].map(
        lambda x: archetype_names.get(int(x), f"Arquetipo {x}")
    )
 
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
 
    df_out.to_parquet(ARTIFACTS_PATH / "base_con_arquetipos.parquet", index=False)
    log.info("Base con arquetipos guardada.")
 
    # Métricas clustering
    filas = []
    for k, v in ward_res.items():
        filas.append({"metodo": "Ward", "k": k, "sil": v["sil"], "db": v["db"],
                      "dunn": v["dunn"], "cv": v["bal"]["cv"],
                      "min_pct": v["bal"]["min_pct"], "score": v["score"]})
    for k, v in kmeans_res.items():
        filas.append({"metodo": "KMeans", "k": k, "sil": v["sil"], "db": v["db"],
                      "dunn": v["dunn"], "cv": v["bal"]["cv"],
                      "min_pct": v["bal"]["min_pct"], "score": v["score"]})
    pd.DataFrame(filas).to_csv(ARTIFACTS_PATH / "metricas_clustering.csv", index=False)
 
    # Cargas factoriales por bloque
    for nombre, info in resumen_afe.items():
        if "cargas" in info:
            info["cargas"].to_csv(ARTIFACTS_PATH / f"cargas_{nombre}.csv")
 
    # Modelo
    modelo = {
        "mca": mca, "qt": qt, "pca": pca_model,
        "k_opt_ward": k_opt_w, "k_sec_ward": k_sec_w,
        "k_opt_kmeans": k_opt_km, "k_sec_kmeans": k_sec_km,
        "archetype_names": archetype_names,
        "bloques_afe": BLOQUES_AFE, "resumen_afe": {
            k: {kk: vv for kk, vv in v.items() if kk != "cargas"}
            for k, v in resumen_afe.items()
        },
        "dbscan_viable": bool(dbscan_res),
    }
    with open(MODELS_PATH / "modelo_arquetipos.pkl", "wb") as f:
        pickle.dump(modelo, f)
    log.info("Modelo guardado.")
 
 
# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
 
def run() -> pd.DataFrame:
    log.info("=" * 55)
    log.info("PIPELINE TRAIN V4 — AFE BLOQUES + CLUSTERING MÚLTIPLE")
    log.info("=" * 55)
 
    df = cargar_base()
 
    # ── ETAPA 1: AFE POR BLOQUES ──────────────────────────────────
    scores_afe, resumen_afe = afe_por_bloques(df)
    imprimir_resumen_afe(resumen_afe)
 
    # ── ETAPA 2: ACM ──────────────────────────────────────────────
    coords_acm, mca, inercia_acm = aplicar_acm(df)
    log.info(f"  ACM inercia: {inercia_acm}")
 
    # ── ETAPA 3: ESPACIO LATENTE + PCA ───────────────────────────
    log.info("=" * 55)
    log.info("ETAPA 3 — ESPACIO LATENTE + PCA (3 componentes)")
    log.info("=" * 55)
    X_latente, qt, pca_model, cols_latente = construir_espacio_latente(
        scores_afe, coords_acm, n_components=3
    )
    log.info(f"  Dimensiones finales: {X_latente.shape}")
 
    # ── ETAPA 4: CLUSTERING ───────────────────────────────────────
    log.info("=" * 55)
    log.info("ETAPA 4 — EVALUACIÓN DE CLUSTERING")
    log.info("=" * 55)
 
    kmeans_res = evaluar_kmeans(X_latente)
    ward_res   = evaluar_ward(X_latente)
    dbscan_res = evaluar_dbscan(X_latente)
 
    # Selección k óptimo
    k_opt_km, k_sec_km = seleccionar_mejor_k(kmeans_res, "KMeans", k_min=3)
    k_opt_w,  k_sec_w  = seleccionar_mejor_k(ward_res,   "Ward",   k_min=3)
 
    imprimir_comparacion_v2(kmeans_res, ward_res, dbscan_res,
                             k_opt_km, k_sec_km, k_opt_w, k_sec_w)
 
    # Labels finales — KMeans es el método principal
    labels_km_opt  = renombrar_por_bienestar(kmeans_res[k_opt_km]["labels"], df)
    labels_km_sec  = renombrar_por_bienestar(kmeans_res[k_sec_km]["labels"], df)
    labels_ward_opt = renombrar_por_bienestar(ward_res[k_opt_w]["labels"], df)
    labels_ward_sec = renombrar_por_bienestar(ward_res[k_sec_w]["labels"], df)
 
    # Distribuciones finales
    print(f"\n{'='*65}")
    print("  DISTRIBUCIÓN FINAL DE ARQUETIPOS")
    print(f"{'='*65}")
    for nombre, labels, k in [
        (f"KMeans k={k_opt_km} (óptimo — método principal)", labels_km_opt,  k_opt_km),
        (f"KMeans k={k_sec_km} (2do mejor)",                  labels_km_sec,  k_sec_km),
        (f"Ward k={k_opt_w} (validación)",                    labels_ward_opt, k_opt_w),
    ]:
        counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)
        print(f"\n  {nombre}:")
        for arq, n in counts.items():
            barra = "█" * int(n/total*30)
            print(f"    {arq}: {n:4d} ({n/total*100:.1f}%) {barra}")
 
    # ── GUARDADO ──────────────────────────────────────────────────
    guardar_todo(df,
                 labels_km_opt, labels_km_sec,
                 labels_ward_opt, labels_ward_sec,
                 k_opt_km, k_sec_km, k_opt_w, k_sec_w,
                 kmeans_res, ward_res, dbscan_res,
                 resumen_afe, mca, qt, pca_model)
 
    log.info("Pipeline completado.")
    return df
 
 
if __name__ == "__main__":
    run()