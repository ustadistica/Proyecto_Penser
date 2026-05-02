<div align="center">

# Proyecto PENSER — Del Índice a los Arquetipos

**Consultorio de Estadística y Ciencia de Datos**  
Universidad Santo Tomás · Facultad de Estadística  
Curso: Consultoría e Investigación · Octavo Semestre (2026-1)

**Equipo:** Haider Rojas · Sergio Prieto

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Tests](https://img.shields.io/badge/Tests-40%20passed-brightgreen)
![Status](https://img.shields.io/badge/Estado-Completado-success)

</div>

---

## ¿De qué trata este proyecto?

Los graduados de la Universidad Santo Tomás no son todos iguales. Este proyecto analiza **dos bases de datos independientes** del Estudio PENSER y construye **arquetipos de graduados** mediante métodos estadísticos avanzados, reemplazando un índice unidimensional no reproducible por un modelo robusto, interpretable y metodológicamente riguroso.

---

## Bases de Datos Analizadas

| Base | Registros | Columnas útiles | Instrumento |
|---|---|---|---|
| **Estudio de Percepción Egresados** | 2.530 | 109 | 23 competencias Likert 1–5, bienestar, satisfacción |
| **Base Depurada PENSER 2025** | 1.129 | 64 | 15 logro competencias + 6 incidencia bienestar |

> **Nota:** Ambas bases pueden contener graduados en común. No es posible verificarlo sin un identificador único compartido entre estudios.

---

## Resultados — Base Percepción Egresados (Ward k=3)

| # | Arquetipo | n | % | Perfil |
|---|---|---|---|---|
| 0 | **El Subjetivamente Satisfecho** | 534 | 21.1% | F1+F2+F3 bajos. 31.8% recomendaría la USTA. |
| 1 | **El Profesional Consolidado** | 1.343 | 53.1% | F1+F2+F3 medios. 82.1% recomendaría la USTA. |
| 2 | **El Líder de Alto Desempeño** | 653 | 25.8% | F1+F2+F3 altos. 87.9% recomendaría la USTA. |

### Factores AFE por Bloques — Base Percepción

| Bloque | Variables | KMO | Factores | Varianza |
|---|---|---|---|---|
| B1 Cognitivo-Comunicativo | 8 | 0.924 Excelente | 1 | 60.8% |
| B2 Tecnológico-Inserción | 7 | 0.869 Bueno | 1 | 64.2% |
| B3 Liderazgo-Social | 7 | 0.905 Excelente | 1 | 62.1% |
| B4a Satisfacción Vital | 3 | 0.644 Mediocre | 1 | 71.7% |
| B4b Correspondencia Laboral | 2 | 0.500 Mediocre | 1 | 100.0% |
| B5 Bienestar | 7 | 0.757 Aceptable | 2 | 49.2% |

> **7 indicadores AFE + 3 dimensiones ACM = 10 dimensiones en espacio latente**

### Hallazgos clave

- **Segunda lengua**: media 2.76/5 — brecha transversal en los 3 arquetipos (43% calificó 1 o 2)
- **Hipótesis del líder más antiguo**: NO confirmada — diferencias de edad no significativas (KW p=0.47)
- **Movilidad social**: 20.7% ascendió de estrato tras graduarse
- **Arquetipo 0**: 31.3% no estudiaría de nuevo en la USTA

---

## Resultados — Base Depurada PENSER 2025 (Ward k=4)

| # | Arquetipo | n | % | Score incidencia |
|---|---|---|---|---|
| 0 | **El Graduado en Desarrollo** | 335 | 29.7% | 8.59/30 |
| 1 | **El Profesional en Formación** | 214 | 19.0% | 19.67/30 |
| 2 | **El Profesional Impactado** | 341 | 30.2% | 20.22/30 |
| 3 | **El Líder con Alta Incidencia** | 239 | 21.2% | 22.39/30 |

### Factores AFE por Bloques — Base Depurada

| Bloque | Variables | KMO | Factores | Varianza |
|---|---|---|---|---|
| B1 Competencias Transversales | 10 | 0.932 Excelente | 1 | 72.5% |
| B2 Competencias Profesionales | 5 | 0.851 Bueno | 1 | 63.0% |
| B3 Incidencia Bienestar | 6 | 0.864 Bueno | 1 | 68.5% |

> **3 indicadores AFE + 3 dimensiones ACM = 6 dimensiones en espacio latente**

### Limitación documentada

Los arquetipos de esta base están fuertemente asociados con la sede de graduación (Bucaramanga → 86.8% Profesional Impactado · Villavicencio → 85.5% Profesional en Formación). El clustering captura diferencias inter-sede más que perfiles puramente individuales.

---

## Metodología

### Etapa 1 — Ingesta y Validación (`ingest.py`)
Pipeline de carga con reporte automático de calidad: PII, duplicados, fila fantasma, outliers Likert, columnas vacías. Guarda en formato Parquet.

### Etapa 2 — Construcción de Variables (`features.py`)
Codificación de escalas ordinales, binarias, categóricas. Construcción de scores compuestos y variables para ACM.

### Etapa 3 — Modelamiento (`train.py`)

**AFE por Bloques Temáticos**
- Variables Likert NO son normales → correlación de **Spearman** (no Pearson)
- KMO individual por bloque — factibilidad verificada antes de factorizar
- Rotación **Varimax** para 1 factor · **Oblimin** para ≥2 factores
- Cada bloque genera sus propios indicadores latentes

**ACM sobre variables categóricas nominales**
- 3 dimensiones retenidas en ambas bases

**Clustering — 3 métodos evaluados**

| Método | Implementación | Resultado |
|---|---|---|
| Ward jerárquico | Espacio latente AFE+ACM estandarizado | **Principal** |
| K-Prototypes | Variables originales (mixtas) | Validación cruzada |
| DBSCAN | Espacio latente | No viable — datos Likert sin densidad |

**Validación con 4 métricas:**
- Silueta (separación)
- Índice de Dunn (inter/intra cluster)
- Davies-Bouldin (compacidad)
- Balance (CV y min% por grupo)

**Score compuesto:** Silueta(35%) + Dunn(25%) + DB⁻¹(25%) + Balance(15%)

**Selección:** k óptimo (mejor score con min%≥15%) + segundo mejor k (mejor score global diferente al óptimo)

### Etapa 4 — Evaluación (`evaluate.py`)
Descriptivas profundas con tests Kruskal-Wallis, análisis por programa académico, edad, año de graduación, trayectoria laboral, movilidad social y comparación entre bases.

---

## Estructura del Proyecto

```
Proyecto_Penser/
│
├── data/
│   ├── raw/          ← Bases originales (no versionadas)
│   ├── interim/      ← Bases validadas (no versionadas)
│   └── processed/    ← Bases procesadas (no versionadas)
│
├── src/
│   ├── percepcion/          ← Pipeline base percepción egresados
│   │   ├── ingest.py        (388 líneas)
│   │   ├── features.py      (584 líneas)
│   │   ├── train.py         (692 líneas) ← V4
│   │   └── evaluate.py      (712 líneas) ← V4
│   └── depurada/            ← Pipeline base depurada PENSER 2025
│       ├── ingest_depurada.py   (317 líneas)
│       ├── features_depurada.py (272 líneas)
│       ├── train_depurada.py    (564 líneas) ← V4
│       └── evaluate_depurada.py (550 líneas) ← V4
│
├── tests/
│   ├── test_ingest.py    ← 14 pruebas
│   ├── test_features.py  ← 12 pruebas
│   └── test_train.py     ← 14 pruebas (V4)
│
├── artifacts/        ← Artefactos generados (no versionados)
├── models/           ← Modelos serializados (no versionados)
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cómo Reproducir el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/ustadistica/Proyecto_Penser.git
cd Proyecto_Penser
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Parche requerido (compatibilidad factor_analyzer con sklearn 1.8)

```bash
FA_PATH=$(python3 -c "import factor_analyzer, os; print(os.path.dirname(factor_analyzer.__file__))")/factor_analyzer.py
sed -i 's/force_all_finite="allow-nan"/ensure_all_finite="allow-nan"/g' $FA_PATH
sed -i 's/force_all_finite=True/ensure_all_finite=True/g' $FA_PATH
```

### 4. Agregar las bases de datos

```
data/raw/ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx
data/raw/DATA_DEPURADA_PENSER_2025.xlsx
```

### 5. Ejecutar pipeline — Base Percepción

```bash
python src/percepcion/ingest.py
python src/percepcion/features.py
python src/percepcion/train.py
python src/percepcion/evaluate.py
```

### 6. Ejecutar pipeline — Base Depurada

```bash
python src/depurada/ingest_depurada.py
python src/depurada/features_depurada.py
python src/depurada/train_depurada.py
python src/depurada/evaluate_depurada.py
```

### 7. Pruebas unitarias

```bash
pytest tests/ -v
# 40 passed
```

---

## Stack Tecnológico

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python 3.12 |
| Manipulación de datos | pandas, numpy |
| Machine Learning | scikit-learn (Ward, DBSCAN, StandardScaler) |
| Clustering mixto | kmodes (KPrototypes) |
| Análisis Factorial | factor_analyzer (AFE por bloques, KMO, Bartlett) |
| Análisis Correspondencias | prince (MCA) |
| Estadística | scipy (Kruskal-Wallis, tests no paramétricos) |
| Calidad de código | pytest (40 pruebas unitarias) |
| Entorno | GitHub Codespaces |
| Formato de datos | Parquet |

---

<div align="center">

Universidad Santo Tomás · Facultad de Estadística · 2026-1  
Consultorio de Estadística y Ciencia de Datos

</div>