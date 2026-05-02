"""
train_depurada.py — Versión 4.0
================================
AFE por Bloques + ACM + Clustering múltiple con validación robusta
Base: DATA DEPURADA PENSER USTA 2025

Autores : Haider Rojas · Sergio Prieto
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2026-1

Bloques AFE:
------------
B1 — Competencias Transversales  (KMO=0.932 Excelente) → 1 factor
B2 — Competencias Profesionales  (KMO=0.851 Bueno)     → 1 factor
B3 — Incidencia en Bienestar     (KMO=0.864 Bueno)     → 1 factor

Espacio latente: 3 factores AFE + 3 dims ACM = 6 dimensiones
"""

import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import prince
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.metrics import (davies_bouldin_score, silhouette_score,
                              pairwise_distances)
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

# ---------------------------------------------------------------------------
# BLOQUES AFE
# ---------------------------------------------------------------------------
BLOQUES_AFE = {
    "B1_Competencias_Transversales": [
        "logro_comunicacion", "logro_relaciones_interpersonales",
        "logro_toma_decisiones", "logro_solucion_problemas",
        "logro_pensamiento_creativo", "logro_pensamiento_critico",
        "logro_manejo_emociones", "logro_manejo_estres",
        "logro_multiculturales", "logro_interculturales",
    ],
    "B2_Competencias_Profesionales": [
        "logro_comp_prof_1", "logro_comp_prof_2", "logro_comp_prof_3",
        "logro_comp_prof_4", "logro_comp_prof_5",
    ],
    "B3_Incidencia_Bienestar": [
        "incidencia_ingresos", "incidencia_empleo", "incidencia_vivienda",
        "incidencia_salud", "incidencia_recreacion", "incidencia_educacion",
    ],
}

N_FACTORES_BLOQUE = {
    "B1_Competencias_Transversales":  1,
    "B2_Competencias_Profesionales":  1,
    "B3_Incidencia_Bienestar":        1,
}

COLS_ACM = ["cat_sede", "cat_programa", "cat_tipo_cargo", "cat_laborando"]

COLS_CATEGORICAS_KPROTO = ["cat_sede", "cat_laborando"]

ARCHETYPE_NAMES_K3 = {
    0: "El Graduado en Desarrollo",
    1: "El Profesional Impactado",
    2: "El Líder con Alta Incidencia",
}

ARCHETYPE_NAMES_K4 = {
    0: "El Graduado en Desarrollo",
    1: "El Profesional en Formación",
    2: "El Profesional Impactado",
    3: "El Líder con Alta Incidencia",
}


# ---------------------------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------------------------

def dunn_index(X: np.ndarray, labels: np.ndarray, n_sample: int = 800) -> float:
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
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    return {
        "cv":      round(float(counts.std() / counts.mean()), 4),
        "min_pct": round(float(counts.min() / total * 100), 2),
        "dist":    dict(counts),
    }


def score_compuesto(sil, db, dunn, cv) -> float:
    return sil * 0.35 + dunn * 0.25 + (1 / (1 + db)) * 0.25 + (1 / (1 + cv)) * 0.15


# ---------------------------------------------------------------------------
# ETAPA 1: AFE POR BLOQUES
# ---------------------------------------------------------------------------

def cargar_base():
    ruta = PROCESSED_PATH / "base_depurada_procesada.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encontrado: {ruta}\n"
            f"Ejecuta: python src/depurada/features_depurada.py"
        )
    df = pd.read_parquet(ruta)
    log.info(f"Base cargada: {df.shape[0]:,} × {df.shape[1]}")
    return df


def afe_por_bloques(df: pd.DataFrame) -> tuple:
    imp = SimpleImputer(strategy="median")
    scores_bloques = {}
    resumen_bloques = {}

    log.info("=" * 55)
    log.info("ETAPA 1 — AFE POR BLOQUES")
    log.info("=" * 55)

    for nombre, cols in BLOQUES_AFE.items():
        cols_ok = [c for c in cols if c in df.columns]
        if not cols_ok:
            log.warning(f"  {nombre}: sin columnas")
            continue

        X = df[cols_ok].apply(pd.to_numeric, errors="coerce")
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=cols_ok, index=df.index)
        X_ranks = X_imp.rank()

        try:
            chi2, p = calculate_bartlett_sphericity(X_ranks)
            _, kmo = calculate_kmo(X_ranks)
        except Exception:
            kmo, p = 0.0, 1.0

        kmo_label = ("Excelente" if kmo >= 0.9 else "Bueno" if kmo >= 0.8
                     else "Aceptable" if kmo >= 0.7 else "Mediocre")
        n_factores = N_FACTORES_BLOQUE.get(nombre, 1)

        try:
            fa = FactorAnalyzer(
                n_factors=n_factores,
                rotation="varimax" if n_factores == 1 else "oblimin",
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
            log.warning(f"  {nombre}: ERROR — {e}")
            continue

    scores_df = pd.DataFrame(scores_bloques, index=df.index)
    log.info(f"  → {scores_df.shape[1]} indicadores AFE generados")
    return scores_df, resumen_bloques


def imprimir_resumen_afe(resumen: dict) -> None:
    print("\n" + "=" * 65)
    print("  AFE POR BLOQUES — BASE DEPURADA")
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
            print(f"    Cargas principales (|>0.5|):")
            for factor in info["cargas"].columns:
                top = info["cargas"][factor].abs()
                top = top[top > 0.5].sort_values(ascending=False)
                if not top.empty:
                    vars_str = ", ".join([
                        f"{v}({info['cargas'].loc[v,factor]:+.2f})"
                        for v in top.index[:3]
                    ])
                    print(f"      {factor}: {vars_str}")


# ---------------------------------------------------------------------------
# ETAPA 2: ACM
# ---------------------------------------------------------------------------

def aplicar_acm(df: pd.DataFrame) -> tuple:
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

def construir_espacio_latente(scores_afe, coords_acm):
    X = pd.concat([scores_afe, coords_acm], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"  Espacio latente: {X_scaled.shape[1]} dims "
             f"({scores_afe.shape[1]} AFE + {coords_acm.shape[1]} ACM)")
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# ETAPA 4: CLUSTERING
# ---------------------------------------------------------------------------

def evaluar_ward(X, k_range=range(2, 8)):
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
            "labels": labels, "sil": round(sil,4), "db": round(db,4),
            "dunn": round(dunn,4), "bal": bal, "score": round(sc,4)
        }
        log.info(f"  k={k}: sil={sil:.4f} | Dunn={dunn:.4f} | DB={db:.4f} | "
                 f"CV={bal['cv']:.3f} | min={bal['min_pct']:.1f}% | score={sc:.4f}")
    return resultados


def evaluar_kprototypes(df, X_latente, k_range=range(2, 8)):
    resultados = {}
    log.info("K-Prototypes...")

    all_num = []
    for cols in BLOQUES_AFE.values():
        all_num += [c for c in cols if c in df.columns]

    cat_cols_kp = [c for c in COLS_CATEGORICAS_KPROTO if c in df.columns]
    imp = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(
        imp.fit_transform(df[all_num].apply(pd.to_numeric, errors="coerce")),
        columns=all_num
    )
    X_cat_kp = df[cat_cols_kp].fillna("Otro").astype(str)
    X_kp = np.hstack([X_num.values, X_cat_kp.values])
    cat_idx = list(range(len(all_num), len(all_num) + len(cat_cols_kp)))

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
                "labels": labels, "sil": round(sil,4), "db": round(db,4),
                "dunn": round(dunn,4), "bal": bal, "score": round(sc,4)
            }
            log.info(f"  k={k}: sil={sil:.4f} | Dunn={dunn:.4f} | DB={db:.4f} | "
                     f"CV={bal['cv']:.3f} | min={bal['min_pct']:.1f}% | score={sc:.4f}")
        except Exception as e:
            log.warning(f"  k={k}: ERROR — {e}")
    return resultados


def evaluar_dbscan(X):
    resultados = {}
    log.info("DBSCAN...")
    for eps in [0.3, 0.5, 0.8, 1.0, 1.5]:
        for min_s in [20, 30]:
            dbs = DBSCAN(eps=eps, min_samples=min_s)
            labels = dbs.fit_predict(X)
            n_cl = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(labels) * 100
            if n_cl >= 2 and noise_pct < 20:
                mask = labels != -1
                sil = silhouette_score(X[mask], labels[mask])
                bal = balance_score(labels[labels != -1])
                resultados[f"eps{eps}_min{min_s}"] = {
                    "n_clusters": n_cl, "noise_pct": round(noise_pct,1),
                    "sil": round(sil,4), "bal": bal
                }
                log.info(f"  eps={eps} min={min_s}: k={n_cl} | "
                         f"ruido={noise_pct:.1f}% | sil={sil:.4f}")
    if not resultados:
        log.info("  DBSCAN: no viable — datos ordinales sin estructura de densidad")
    return resultados


def seleccionar_mejor_k(resultados, metodo, k_min=3):
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

    candidatos_sec = {k: v for k, v in resultados.items() if k != k_opt}
    if candidatos_sec:
        k_second = sorted(candidatos_sec.items(), key=lambda x: x[1]["score"], reverse=True)[0][0]
    else:
        k_second = k_opt

    log.info(f"  {metodo} → óptimo=k{k_opt} "
             f"(score={resultados[k_opt]['score']:.4f}, min%={resultados[k_opt]['bal']['min_pct']:.1f}%) | "
             f"2do=k{k_second} "
             f"(score={resultados[k_second]['score']:.4f}, min%={resultados[k_second]['bal']['min_pct']:.1f}%)")
    return k_opt, k_second


def renombrar_por_incidencia(labels, df):
    col = "score_impacto_formacion"
    if col not in df.columns:
        return labels
    df_t = df.reset_index(drop=True).copy()
    df_t["_cl"] = labels
    orden = df_t.groupby("_cl")[col].mean().sort_values().index.tolist()
    mapa = {old: new for new, old in enumerate(orden)}
    log.info(f"  Renombrado por incidencia: {mapa}")
    return np.array([mapa[l] for l in labels])


# ---------------------------------------------------------------------------
# IMPRIMIR Y GUARDAR
# ---------------------------------------------------------------------------

def imprimir_comparacion(ward_res, kproto_res, dbscan_res,
                          k_opt_w, k_sec_w, k_opt_kp, k_sec_kp):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  COMPARACIÓN DE MÉTODOS — BASE DEPURADA")
    print(sep)
    print(f"\n{'MÉTODO':<15} {'k':>3} {'Silueta':>9} {'Dunn':>9} {'DB':>9} "
          f"{'CV':>8} {'min%':>7} {'score':>8}")
    print("-" * 65)
    for k, v in sorted(ward_res.items()):
        marca = "← ÓPT" if k==k_opt_w else ("← 2do" if k==k_sec_w else "")
        print(f"{'Ward':<15} {k:>3} {v['sil']:>9.4f} {v['dunn']:>9.4f} "
              f"{v['db']:>9.4f} {v['bal']['cv']:>8.3f} {v['bal']['min_pct']:>7.1f}% "
              f"{v['score']:>8.4f} {marca}")
    print()
    for k, v in sorted(kproto_res.items()):
        marca = "← ÓPT" if k==k_opt_kp else ("← 2do" if k==k_sec_kp else "")
        print(f"{'KPrototypes':<15} {k:>3} {v['sil']:>9.4f} {v['dunn']:>9.4f} "
              f"{v['db']:>9.4f} {v['bal']['cv']:>8.3f} {v['bal']['min_pct']:>7.1f}% "
              f"{v['score']:>8.4f} {marca}")
    if dbscan_res:
        print("\nDBSCAN viables:")
        for cfg, v in dbscan_res.items():
            print(f"  {cfg}: k={v['n_clusters']} | ruido={v['noise_pct']}% | sil={v['sil']:.4f}")
    else:
        print("\nDBSCAN: No viable")
    print(f"\n{sep}")


def guardar_todo(df, labels_w_opt, labels_w_sec, labels_kp_opt, labels_kp_sec,
                 k_opt_w, k_sec_w, k_opt_kp, k_sec_kp,
                 ward_res, kproto_res, resumen_afe, mca, scaler):

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    archetype_names = ARCHETYPE_NAMES_K4 if k_opt_w == 4 else ARCHETYPE_NAMES_K3

    df_out = df.reset_index(drop=True).copy()
    df_out["arquetipo_ward_opt"]   = labels_w_opt
    df_out["arquetipo_ward_sec"]   = labels_w_sec
    df_out["arquetipo_kproto_opt"] = labels_kp_opt
    df_out["arquetipo_kproto_sec"] = labels_kp_sec
    df_out["arquetipo"]            = labels_w_opt
    df_out["nombre_arquetipo"]     = df_out["arquetipo"].map(
        lambda x: archetype_names.get(int(x), f"Arquetipo {x}")
    )
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
    df_out.to_parquet(ARTIFACTS_PATH / "base_depurada_con_arquetipos.parquet", index=False)

    filas = []
    for k, v in ward_res.items():
        filas.append({"metodo":"Ward","k":k,"sil":v["sil"],"db":v["db"],
                      "dunn":v["dunn"],"cv":v["bal"]["cv"],"min_pct":v["bal"]["min_pct"],"score":v["score"]})
    for k, v in kproto_res.items():
        filas.append({"metodo":"KPrototypes","k":k,"sil":v["sil"],"db":v["db"],
                      "dunn":v["dunn"],"cv":v["bal"]["cv"],"min_pct":v["bal"]["min_pct"],"score":v["score"]})
    pd.DataFrame(filas).to_csv(ARTIFACTS_PATH / "metricas_clustering_depurada.csv", index=False)

    for nombre, info in resumen_afe.items():
        if "cargas" in info:
            info["cargas"].to_csv(ARTIFACTS_PATH / f"cargas_depurada_{nombre}.csv")

    with open(MODELS_PATH / "modelo_depurada.pkl", "wb") as f:
        pickle.dump({
            "mca": mca, "scaler": scaler,
            "k_opt_ward": k_opt_w, "k_sec_ward": k_sec_w,
            "k_opt_kproto": k_opt_kp, "k_sec_kproto": k_sec_kp,
            "archetype_names": archetype_names,
            "bloques_afe": BLOQUES_AFE,
        }, f)
    log.info("Modelo depurada guardado.")


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def run():
    log.info("=" * 55)
    log.info("PIPELINE TRAIN DEPURADA V4")
    log.info("=" * 55)

    df = cargar_base()

    scores_afe, resumen_afe = afe_por_bloques(df)
    imprimir_resumen_afe(resumen_afe)

    coords_acm, mca, inercia = aplicar_acm(df)

    log.info("=" * 55)
    log.info("ETAPA 3 — ESPACIO LATENTE")
    log.info("=" * 55)
    X_latente, scaler = construir_espacio_latente(scores_afe, coords_acm)

    log.info("=" * 55)
    log.info("ETAPA 4 — CLUSTERING")
    log.info("=" * 55)
    ward_res   = evaluar_ward(X_latente)
    kproto_res = evaluar_kprototypes(df, X_latente)
    dbscan_res = evaluar_dbscan(X_latente)

    k_opt_w,  k_sec_w  = seleccionar_mejor_k(ward_res,   "Ward")
    k_opt_kp, k_sec_kp = seleccionar_mejor_k(kproto_res, "KPrototypes")

    imprimir_comparacion(ward_res, kproto_res, dbscan_res,
                          k_opt_w, k_sec_w, k_opt_kp, k_sec_kp)

    labels_w_opt  = renombrar_por_incidencia(ward_res[k_opt_w]["labels"], df)
    labels_w_sec  = renombrar_por_incidencia(ward_res[k_sec_w]["labels"], df)
    labels_kp_opt = renombrar_por_incidencia(kproto_res[k_opt_kp]["labels"], df)
    labels_kp_sec = renombrar_por_incidencia(kproto_res[k_sec_kp]["labels"], df)

    print(f"\n{'='*65}")
    print("  DISTRIBUCIÓN FINAL")
    print(f"{'='*65}")
    for nombre, labels, k in [
        (f"Ward k={k_opt_w} (óptimo)",   labels_w_opt,  k_opt_w),
        (f"Ward k={k_sec_w} (2do mejor)", labels_w_sec,  k_sec_w),
        (f"KProto k={k_opt_kp} (óptimo)", labels_kp_opt, k_opt_kp),
        (f"KProto k={k_sec_kp} (2do)",    labels_kp_sec, k_sec_kp),
    ]:
        counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)
        print(f"\n  {nombre}:")
        for arq, n in counts.items():
            barra = "█" * int(n/total*30)
            print(f"    {arq}: {n:4d} ({n/total*100:.1f}%) {barra}")

    guardar_todo(df, labels_w_opt, labels_w_sec, labels_kp_opt, labels_kp_sec,
                 k_opt_w, k_sec_w, k_opt_kp, k_sec_kp,
                 ward_res, kproto_res, resumen_afe, mca, scaler)

    log.info("Pipeline depurada completado.")
    return df


if __name__ == "__main__":
    run()