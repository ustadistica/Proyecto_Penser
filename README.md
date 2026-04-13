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

El punto de partida fue revisar el Índice de Impacto de Egresados que ya existía, encontrar sus limitaciones, y construir algo más robusto y reproducible.

---

## Resultados

### Los tres arquetipos identificados

| # | Arquetipo | Graduados | % | Perfil |
|---|---|---|---|---|
| 0 | **El Subjetivamente Satisfecho** | 338 | 13.4% | Competencias en desarrollo (media 2.9/5). Mayor área de mejora institucional. |
| 1 | **El Profesional Consolidado** | 1.256 | 49.6% | Perfil más común. Competencias medias-altas (media 3.8/5). Sólido y equilibrado. |
| 2 | **El Líder de Alto Desempeño** | 936 | 37.0% | Competencias muy altas (media 4.6/5). Referente del impacto de la formación USTA. |

### Hallazgos clave

**Segunda lengua** es la brecha más crítica y transversal — con una media de **2.76/5**, el 43% de los graduados la calificó con 1 o 2, sin importar el arquetipo al que pertenecen. Es la recomendación institucional más directa del estudio.

**El 20.7%** de los graduados ascendió de estrato socioeconómico tras obtener su título — evidencia concreta del impacto de la formación en movilidad social.

**El score promedio de bienestar** es de 3.96/7, con diferencias claras entre arquetipos: el Líder de Alto Desempeño alcanza 4.65/7 frente al 3.15/7 del Subjetivamente Satisfecho.

---

## Datos y Calidad

La base cruda del Estudio PENSER 2026 tenía **2.596 registros y 157 columnas**. Tras el pipeline de limpieza quedaron **2.530 registros y 109 columnas** — una pérdida del 2.5%, completamente justificada:

| Problema detectado | Registros eliminados | Justificación |
|---|---|---|
| Filas completamente vacías | 2 | Sin ninguna respuesta — no representan graduados reales |
| Fila fantasma | 1 | Contenía números de pregunta (87–109) en lugar de respuestas |
| Duplicados exactos | 48 | Formularios enviados más de una vez por el mismo graduado |
| Formularios incompletos | 15 | Respondieron menos del 10% del formulario (promedio: 3.5 de 157 preguntas) |
| **Total eliminados** | **66 (2.5%)** | |

En columnas:

| Tipo de columna eliminada | Columnas | Justificación |
|---|---|---|
| Índice oculto del Excel | 1 | Artefacto de exportación, no es una variable |
| Datos personales (PII) | 2 | Nombre y email — privacidad de los graduados |
| Columnas con >95% nulos | 45 | Programas de sedes y ciudades casi sin respuestas |
| **Total eliminadas** | **48** | |

---

## Metodología

### Etapa 1 — Ingesta y Validación (`ingest.py`)

Pipeline de carga con reporte automático de calidad. Detecta y documenta cada problema encontrado con logs con timestamp. Guarda la base limpia en formato Parquet para eficiencia.

### Etapa 2 — Construcción de Variables (`features.py`)

Transforma la base limpia en variables analíticas:

- **23 competencias** Likert 1–5 con nombres estandarizados
- **7 indicadores de bienestar** codificados de Sí/No a 0/1
- **Índice de movilidad social**: estrato actual − estrato al graduarse
- **Score de bienestar**: suma de los 7 indicadores (rango 0–7)
- **5 escalas de satisfacción** convertidas de texto a numérico
- **Variables sociodemográficas**: género, sede y nivel de formación

### Etapa 3 — Modelamiento (`train.py`)

**¿Por qué PCA y no Análisis Factorial Exploratorio clásico?**
Las 23 competencias tienen una correlación media de 0.87 entre sí — están tan relacionadas que un solo componente explica el 37.3% de la varianza. Esta estructura no es compatible con el AFE clásico, que requiere dimensiones más independientes. El PCA con 18 componentes explica el 85.4% de la varianza y reduce el ruido antes del clustering.

**¿Por qué k=3 y no k=5 como el índice original?**
Los datos sugieren 3 grupos óptimos según el coeficiente de silueta (0.1685 para k=3 vs 0.1262 para k=5). Los graduados de la USTA comparten una formación común que produce perfiles similares. Forzar k=5 generaría arquetipos artificialmente fragmentados sin sustento estadístico.

Se evaluaron **KMeans** y **Clustering Jerárquico (Ward)** con k=2 a 7, usando tres métricas de validación simultáneas: coeficiente de silueta, Davies-Bouldin y Calinski-Harabasz.

### Etapa 4 — Evaluación e Interpretación (`evaluate.py`)

Descripción detallada de cada arquetipo con fortalezas, brechas, distribución de bienestar, movilidad social y recomendaciones institucionales.

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
│   ├── 01_eda.ipynb          ← Análisis exploratorio completo con visualizaciones
│   └── 02_modeling.ipynb     ← PCA, clustering y perfil de arquetipos
│
├── src/
│   ├── __init__.py
│   ├── ingest.py     ← Carga, limpieza y reporte de calidad
│   ├── features.py   ← Construcción de variables analíticas
│   ├── train.py      ← PCA + KMeans/Jerárquico + selección de k óptimo
│   └── evaluate.py   ← Descripción e interpretación de arquetipos
│
├── models/           ← Modelo serializado (no versionado — derivado de datos privados)
├── artifacts/        ← Artefactos generados (no versionados — derivado de datos privados)
│
├── tests/
│   ├── test_ingest.py    ← 14 pruebas unitarias
│   ├── test_features.py  ← 11 pruebas unitarias
│   └── test_train.py     ← 11 pruebas unitarias
│
├── .gitignore
└── README.md
```

> **Nota sobre privacidad:** Los datos del Estudio PENSER no están en el repositorio por protección de la información de los graduados. El `.gitignore` excluye toda la carpeta `data/`, `models/` y `artifacts/`.

---

## Cómo Reproducir el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/haiderrojassalazar089-bot/Proyecto_Penser.git
cd Proyecto_Penser
```

### 2. Instalar dependencias

```bash
pip install pandas openpyxl pyarrow scikit-learn matplotlib seaborn scipy pytest
```

### 3. Agregar la base de datos

Coloca el archivo Excel en:
```
data/raw/ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx
```

### 4. Ejecutar el pipeline

```bash
python src/ingest.py      # Limpieza y validación
python src/features.py    # Variables analíticas
python src/train.py       # PCA y clustering
python src/evaluate.py    # Reporte de arquetipos
```

### 5. Verificar con pruebas unitarias

```bash
pytest tests/ -v
# 36 passed
```

### 6. Explorar los notebooks

```
notebooks/01_eda.ipynb       ← Distribuciones, correlaciones, bienestar
notebooks/02_modeling.ipynb  ← PCA, dendrograma, radar chart, perfil arquetipos
```

---

## Stack Tecnológico

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python 3.12 |
| Manipulación de datos | pandas, numpy |
| Machine Learning | scikit-learn (PCA, KMeans, AgglomerativeClustering) |
| Visualización | matplotlib, seaborn |
| Estadística | scipy (linkage, dendrogram) |
| Calidad de código | pytest (36 pruebas unitarias) |
| Entorno de desarrollo | GitHub Codespaces |
| Formato de datos | Parquet |

---

<div align="center">

Universidad Santo Tomás · Facultad de Estadística · 2026-1  
Consultorio de Estadística y Ciencia de Datos

</div>