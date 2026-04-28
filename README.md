<div align="center">

# Proyecto PENSER — Del Índice a los Arquetipos

**Consultorio de Estadística y Ciencia de Datos**  
Universidad Santo Tomás · Facultad de Estadística  
Curso: Consultoría e Investigación · Octavo Semestre (2026-1)

**Equipo:** Haider Rojas · Sergio Prieto

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Tests](https://img.shields.io/badge/Tests-36%20passed-brightgreen)
![Status](https://img.shields.io/badge/Estado-Completado-success)

</div>

---

## ¿De qué trata este proyecto?

Los graduados de la Universidad Santo Tomás no son todos iguales. Unos dominan el liderazgo pero batallan con el inglés. Otros tienen alta satisfacción con su vida pero menor inserción laboral. Otros lo tienen todo.

Este proyecto toma los datos del **Estudio PENSER 2026** — 2.596 graduados encuestados sobre sus competencias, bienestar, trayectoria laboral y calidad de vida — y los convierte en tres **arquetipos** que resumen la diversidad de caminos que toman los egresados de la USTA.

El punto de partida fue revisar el Índice de Impacto de Egresados que ya existía, encontrar sus limitaciones, y construir algo más robusto, reproducible y metodológicamente riguroso.

---

## Resultados

### Los tres arquetipos identificados

| # | Arquetipo | Graduados | % | Perfil |
|---|---|---|---|---|
| 0 | **El Subjetivamente Satisfecho** | 594 | 23.5% | F1+F2+F3 bajos. Solo el 29.5% recomendaría la USTA — señal de alerta institucional. |
| 1 | **El Profesional Consolidado** | 602 | 23.8% | F1+F2+F3 medios. Perfil equilibrado, 90.5% recomendaría la USTA. |
| 2 | **El Líder de Alto Desempeño** | 1.334 | 52.7% | F1+F2+F3 altos. El perfil más frecuente y referente institucional. |

### Factores que sustentan los arquetipos (AFE)

| Factor | Nombre | Varianza | Variables clave |
|---|---|---|---|
| F1 | Competencias Cognitivas y Comunicativas | 21.6% | argumentación, pensamiento crítico, comunicación escrita/oral |
| F2 | Satisfacción y Correspondencia Laboral | 11.2% | satisfacción vida, efecto título, correspondencia empleo |
| F3 | Competencias Tecnológicas e Inserción Laboral | 17.7% | herramientas modernas, inserción laboral, TIC |

### Hallazgos clave

**Segunda lengua** es la brecha más crítica y transversal — media **2.76/5**, el 43% calificó con 1 o 2, sin importar el arquetipo. Es la recomendación institucional más directa del estudio.

**El Arquetipo 0** es la señal de alerta más fuerte: solo el 29.5% recomendaría la USTA y apenas el 18.9% estudiaría otra vez. Los graduados menos satisfechos con su formación no la recomiendan.

**El 20.7%** ascendió de estrato socioeconómico tras graduarse — evidencia concreta del impacto de la formación en movilidad social.

---

## Datos y Calidad

La base cruda tenía **2.596 registros y 157 columnas**. Tras el pipeline de limpieza quedaron **2.530 registros y 109 columnas** — pérdida del 2.5%, completamente justificada:

| Problema detectado | Registros eliminados | Justificación |
|---|---|---|
| Filas completamente vacías | 2 | Sin ninguna respuesta |
| Fila fantasma | 1 | Tenía números de pregunta (87–109) en lugar de respuestas |
| Duplicados exactos | 48 | Formularios enviados más de una vez |
| Formularios incompletos (<10%) | 15 | Promedio: 3.5 respuestas de 157 |
| **Total eliminados** | **66 (2.5%)** | |

| Tipo de columna eliminada | Columnas | Justificación |
|---|---|---|
| Índice oculto del Excel | 1 | Artefacto de exportación |
| Datos personales (PII) | 2 | Nombre y email |
| Columnas con >95% nulos | 45 | Programas de sedes sin respuestas |
| **Total eliminadas** | **48** | |

---

## Metodología

### Etapa 1 — Ingesta y Validación (`ingest.py`)

Pipeline de carga con reporte automático de calidad. Detecta PII, fila fantasma, duplicados, outliers Likert, valores inválidos en binarias y columnas casi vacías. Guarda en formato Parquet.

### Etapa 2 — Construcción de Variables (`features.py`)

- **23 competencias** Likert 1–5 renombradas a nombres cortos
- **15 variables binarias** de bienestar y contexto familiar/laboral
- **6 escalas de satisfacción** convertidas de texto a numérico (0–5)
- **Índice de movilidad social**: estrato actual − estrato al graduarse
- **Nivel de cargo**: recodificado de 119 categorías texto libre a 6 niveles ordinales
- **Salarios**: rangos ordinales unificados por nivel de formación
- **7 variables categóricas** limpias para ACM: género, sede, estado civil, tipo contrato, nivel educativo padres, recomendaría, estudiaría otra vez

### Etapa 3 — Modelamiento (`train.py`)

**Análisis Factorial Exploratorio (AFE)**
- Variables: 23 competencias Likert + 5 escalas de satisfacción (28 variables)
- Las variables Likert NO son normales (Shapiro p=0.000, sesgo negativo) → se usa **correlación de Spearman**
- KMO=0.953 (Excelente) y Bartlett p≈0 confirman factibilidad
- Rotación **Oblimin** — competencias correlacionadas entre sí (r_Spearman=0.50), Varimax asumiría independencia incorrectamente
- **3 factores** por criterio Kaiser (eigenvalor >1): explican el 50.6% de la varianza

**Análisis de Correspondencia Múltiple (ACM)**
- 6 variables categóricas nominales
- 3 dimensiones retenidas (inercia: 37.9% / 31.5% / 30.6%)

**Clustering Jerárquico Ward**
- Input: scores AFE (3) + coordenadas ACM (3) = espacio latente de 6 dimensiones
- Distancia euclidiana sobre espacio latente estandarizado
- k=2 a 7 evaluados con silueta, Davies-Bouldin y Calinski-Harabasz
- **k=3** seleccionado por parsimonia (diferencia silueta k=3 vs k=6 es 0.02; interpretabilidad > optimización marginal)

### Etapa 4 — Evaluación e Interpretación (`evaluate.py`)

Descripción detallada de cada arquetipo con perfil de competencias, bienestar, satisfacción, trayectoria laboral, perfil categórico y recomendaciones institucionales.

---

## Estructura del Proyecto

```
Proyecto_Penser/
│
├── data/
│   ├── raw/          ← Base original (no versionada — privacidad de datos)
│   ├── interim/      ← Base validada por ingest.py (no versionada)
│   └── processed/    ← Base procesada por features.py (no versionada)
│
├── notebooks/
│   ├── 01_eda.ipynb          ← Análisis exploratorio completo
│   └── 02_modeling.ipynb     ← AFE, ACM, clustering y perfil de arquetipos
│
├── src/
│   ├── __init__.py
│   ├── ingest.py     ← Carga, limpieza y reporte de calidad
│   ├── features.py   ← Construcción de variables analíticas y categóricas para ACM
│   ├── train.py      ← AFE (Spearman·Oblimin) + ACM + Ward k=3
│   └── evaluate.py   ← Descripción e interpretación de arquetipos
│
├── models/           ← Modelo serializado (no versionado)
├── artifacts/        ← Artefactos generados (no versionados)
│
├── tests/
│   ├── test_ingest.py    ← 14 pruebas unitarias
│   ├── test_features.py  ← 11 pruebas unitarias
│   └── test_train.py     ← 11 pruebas unitarias
│
├── .gitignore
└── README.md
```

> **Nota sobre privacidad:** Los datos del Estudio PENSER no están en el repositorio. El `.gitignore` excluye `data/`, `models/` y `artifacts/`.

---

## Cómo Reproducir el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/haiderrojassalazar089-bot/Proyecto_Penser.git
cd Proyecto_Penser
```

### 2. Instalar dependencias

```bash
pip install pandas openpyxl pyarrow scikit-learn matplotlib seaborn scipy pytest factor_analyzer prince
```

### 3. Agregar la base de datos

```
data/raw/ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx
```

### 4. Ejecutar el pipeline

```bash
python src/ingest.py      # Limpieza y validación
python src/features.py    # Variables analíticas + categóricas para ACM
python src/train.py       # AFE + ACM + Ward k=3
python src/evaluate.py    # Reporte de arquetipos
```

### 5. Verificar con pruebas unitarias

```bash
pytest tests/ -v
# 36 passed
```

---

## Stack Tecnológico

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python 3.12 |
| Manipulación de datos | pandas, numpy |
| Machine Learning | scikit-learn (AgglomerativeClustering, StandardScaler) |
| Análisis Factorial | factor_analyzer (AFE, KMO, Bartlett) |
| Análisis Correspondencia | prince (MCA) |
| Visualización | matplotlib, seaborn |
| Estadística | scipy |
| Calidad de código | pytest (36 pruebas unitarias) |
| Entorno de desarrollo | GitHub Codespaces |
| Formato de datos | Parquet |

---

<div align="center">

Universidad Santo Tomás · Facultad de Estadística · 2026-1  
Consultorio de Estadística y Ciencia de Datos

</div>