# Resumen del Proyecto PetFinder — Laboratorio de Implementación II
**Universidad Austral · Abril 2026**
**Autores:** Roxana Alberti · Sandra Sschicchi · Fernando Paganini · Baltazar Villanueva · Paula Calviello · Rosana Martinez

---

## ¿Qué problema resolvemos?

Predecir la **velocidad de adopción** de mascotas en Malasia (competencia Kaggle PetFinder 2019).

- **Target:** `AdoptionSpeed` — 5 clases (0 = mismo día, 4 = nunca adoptado en 100 días)
- **Métrica:** Cohen's Kappa cuadrático — penaliza más los errores grandes (confundir clase 0 con clase 4 cuesta más que confundir 0 con 1)
- **Dataset:** ~15.000 mascotas con texto, imágenes, metadatos de sentimiento y visión por computadora

---

## Progresión de resultados (kappa test)

| Notebook | Técnica | Kappa Test | Mejora |
|---|---|---|---|
| nb04 | Baseline (19 features) | 0.3133 | — |
| nb04 | FE v1 (26 feat) | 0.3231 | +0.0099 |
| nb04 | FE v2 + Optuna simple | 0.3371 | +0.0238 |
| nb04 | FE v3 + Optuna simple | 0.3595 | +0.0462 |
| nb05 | **FE v4 + 5-fold CV + Optuna** | **0.3867** | +0.0734 |
| nb06 | Blend LGB + DistilBERT 95/5 | 0.3699 | — (no mejora) |
| nb07 | SHAP selection 25 feat | 0.3738 | — (referencia) |
| nb08 | Blend LGB 50% + XGB 50% | 0.3906 | +0.0773 |
| nb09 | Blend LGB + XGB + BERT 3% | 0.3935 | +0.0802 |
| **nb10** | **Blend LGB(30%)+XGB(45%)+CB(25%)** | **0.3951** | **+0.0818** |

---

## Paso a paso — qué hicimos en cada notebook

### EDA (eda_petfinder.ipynb)
- Análisis exploratorio: distribución del target, correlaciones, análisis de texto y sentimiento
- Visualizaciones interactivas en Plotly Dash (dashboard desplegado en Render)
- Hallazgo clave: los animales con más fotos y descripciones largas se adoptan más rápido

### nb04 — Feature Engineering v1/v2/v3 + Optuna baseline
**Feature Engineering progresivo:**
- **FE v1:** HasPhoto, HasVideo, IsFree, AgeGroup, HealthScore, IsPureBreed, PhotoPerAnimal
- **FE v2:** interacciones (Age × PhotoAmt, IsPureBreed × AgeGroup, etc.)
- **FE v3:** features de texto/imagen (sentiment_score, avg_label_score, desc_length, etc.)

**Problema identificado:** Optuna optimizaba sobre el test set → overfitting. Gap train/test de 0.28.

### nb05 — FE v4 + 5-fold CV + Target Encoding + Optuna
**Cambios clave respecto a nb04:**

1. **Target Encoding** para Breed1 (307 valores) y State (15 valores):
   - Formula: `(count × mean_categoria + smoothing × mean_global) / (count + smoothing)`
   - Smoothing=10: categorías con pocos datos se acercan a la media global
   - Evita el problema de tratar la raza "250" como numéricamente similar a "251"

2. **5-fold Stratified CV con early stopping:**
   - Optuna nunca ve el test set → los hiperparámetros no se sobreajustan al test
   - Early stopping dentro de cada fold → el modelo no sobreajusta el fold de validación

3. **FE v4 — nuevas features sin data leakage:**
   - `rescuer_n_pets`: cuántas mascotas publicó el mismo rescatador (solo conteo, sin target)
   - `age_rel_breed`: edad del animal / mediana de edad de su raza+tipo (captura "cachorro viejo para su raza")
   - NLP básico: word_count, unique_words, avg_word_len, uppercase_ratio, has_exclamation

4. **FE v5 (aporte del compañero):**
   - `HasName`: 1 si el animal tiene nombre asignado (>2 chars) → +0.0067 kappa
   - `AdoptionFriction`: (Age × Health) / (HasName + IsPureBreed + HealthScore + 1) → índice compuesto

**Error importante que encontramos y corregimos — Data Leakage:**
La primera versión de FE v4 calculaba `breed_speed_mean`, `state_speed_mean` y `rescuer_speed_mean` usando el target (AdoptionSpeed) sobre todo el train. Eso produjo CV kappa 0.7685 (falso). Al computar medias del target fuera del fold, el modelo "ve" información del futuro. Solución: usar solo conteos y features sin target.

### nb06 — Transfer Learning: DistilBERT
- Fine-tuning de DistilBERT sobre el campo `Description` (texto libre, mayormente malayo/inglés)
- Entrenado en Google Colab con GPU T4 (~2 horas)
- Resultado: BERT solo no mejora (el texto ya está capturado parcialmente por las features NLP)
- En blend con 3% de peso sí aporta (nb09: +0.0029)

### nb07 — SHAP + Boruta: selección de features

**Boruta:**
- Crea "shadow features" (copias aleatorizadas) y compara importancia real vs. sombras en 50 iteraciones
- Resultado: solo acepta 1 feature (`avg_label_score`), 1 tentativa (`age_rel_breed`), 46 rechazadas
- **Conclusión:** Boruta con Random Forest es muy conservador para este dataset — RF no aprovecha las features tan bien como LightGBM, por eso la mayoría no superan el umbral estadístico. No significa que las demás sean inútiles.

**SHAP (SHapley Additive exPlanations):**
- Mide cuánto cambia la predicción individual cuando se incluye/excluye cada feature (teoría de juegos cooperativos)
- A diferencia del feature importance de LightGBM (frecuencia de splits), SHAP mide impacto real
- Top features: `rescuer_n_pets` (0.278), `age_rel_breed` (0.243), `Breed1_enc` (0.169)
- **Las features nuevas del FE v4 son las más importantes** — confirma que valió la pena agregarlas
- Modelo con top-25 features (SHAP ≥ 0.04): kappa 0.3738 — solo 0.013 menos con la mitad de variables

### nb08 — LightGBM + XGBoost Ensemble
**¿Por qué combinar LGB y XGB si ambos son gradient boosting?**
- LightGBM crece **leaf-wise** (encuentra la hoja de mayor ganancia y la divide)
- XGBoost crece **depth-wise** (divide todos los nodos del mismo nivel)
- Cometen errores en casos distintos → al promediar probabilidades, los errores se cancelan

**Resultado:** Blend 50/50 = 0.3906 (+0.0039 vs LGB solo)

También se corrió **Optuna para XGBoost** (50 trials, sección F del nb08).

### nb09 — Blend Triple LGB + XGB + BERT
- Combina los tres modelos con grid search de pesos
- BERT con solo 3% de peso mejora el blend binario: 0.3935

### nb10 — CatBoost
**CatBoost (Yandex, 2017):**
- Crecimiento **depth-wise simétrico** (todos los nodos del mismo nivel se dividen igual)
- Manejo nativo de variables categóricas (ordered target encoding interno)
- Más conservador que LGB → menos overfitting individual pero también menos preciso solo
- Resultado individual: 0.3489 (bajo)
- En ensemble: aporta diversidad → Blend LGB(30%)+XGB(45%)+CB(25%) = **0.3951 (record)**

---

## Conceptos clave para explicar al profesor

### 1. ¿Por qué 5-fold CV en lugar de train/test simple?
Con un solo split, Optuna puede encontrar parámetros que funcionan bien solo en ese test específico (overfitting al test set). Con 5-fold CV, los hiperparámetros deben funcionar bien en 5 splits distintos → más generalización. El test set **nunca influye** en la búsqueda de hiperparámetros.

### 2. ¿Por qué Target Encoding y no One-Hot Encoding?
Breed1 tiene 307 valores únicos. One-Hot generaría 307 columnas binarias → curse of dimensionality. Target encoding colapsa eso en 1 columna continua que captura información real del target. El smoothing evita que razas con 1-2 ejemplos tengan encodings extremos.

### 3. ¿Qué es SHAP y por qué es mejor que feature importance?
Feature importance de LightGBM cuenta cuántas veces se usó una feature para hacer splits. Una feature puede tener alta frecuencia pero bajo impacto real. SHAP calcula la contribución marginal de cada feature a cada predicción individual (Shapley values de teoría de juegos). Es más costoso pero más honesto.

### 4. ¿Por qué funciona el ensemble?
Un modelo ensemble es mejor que sus componentes si los modelos cometen **errores incorrelacionados**. LGB, XGB y CatBoost tienen distintos sesgos inductivos (leaf-wise vs. depth-wise vs. depth-wise simétrico). Al promediar sus probabilidades, los errores individuales se cancelan parcialmente. El blend no es promedio simple — encontramos los pesos óptimos por grid search.

### 5. ¿Por qué BERT con 3% y no más?
BERT captura información semántica del texto crudo, pero muchas de esas señales ya están capturadas en features NLP (word_count, sentiment_score, avg_label_score). El texto en PetFinder es además muy corto y ruidoso (malayo, inglés, errores). Con más peso, BERT introduce ruido. Al 3% solo aporta la información marginal que los demás no tienen.

### 6. Data Leakage — el error que tuvimos y corregimos
Al calcular `rescuer_speed_mean` (media de AdoptionSpeed por rescatador) sobre todo el train, el fold de validación "sabe" la media del target de ese rescatador → el modelo aprende una correlación falsa. CV kappa subió a 0.7685 (irreal). Esto se detecta cuando CV >> test. Solución: usar solo `rescuer_n_pets` (conteo, sin target).

### 7. ¿Por qué Boruta rechaza casi todo?
Boruta usa Random Forest internamente. RF no aprovecha las features tan bien como LightGBM (no usa early stopping, no hace target encoding, divide features al azar). El umbral estadístico que Boruta usa es muy conservador (95% de confianza en 50 iteraciones). Resultado: solo `avg_label_score` pasa el umbral. Eso no significa que las demás no sean útiles — lo confirma SHAP que las usa con LightGBM.

---

## Arquitectura del dashboard

Dashboard interactivo en Plotly Dash con 5 tabs:
1. **Distribuciones** — análisis univariado del target y variables principales
2. **Correlaciones** — heatmap y scatter plots
3. **Texto & Sentiment** — análisis de descripciones, sentimiento, largo de texto
4. **Significación** — tests estadísticos entre grupos
5. **Modelo** — comparativa de todos los modelos, SHAP importance, evolución del kappa

Desplegado en **Render.com** (https://ua-mdm-labo2.onrender.com/)

---

## Resumen de lo que aporta cada técnica al kappa final

| Técnica | Impacto en kappa | Por qué funciona |
|---|---|---|
| FE v1-v3 (features básicas) | +0.0462 | Captura señales directas (fotos, salud, edad) |
| Target Encoding | +~0.03 | Breed1 con 307 valores → señal real de adopción |
| 5-fold CV (anti-overfitting) | Gap: 0.28 → 0.10 | Parámetros más generalizables |
| FE v4 (rescuer, edad relativa, NLP) | +0.027 | Señales ocultas en el comportamiento del rescatador |
| FE v5 (HasName, AdoptionFriction) | +0.007 | Aporte del compañero — nombre como señal de historia |
| Ensemble LGB+XGB+CB | +0.009 | Diversidad de sesgos entre modelos |
| BERT en blend (3%) | +0.003 | Información semántica marginal del texto |
| **Total vs. baseline** | **+0.082** | |

---

## Archivos del proyecto

```
input/Notebook generados/
├── EDA/
│   ├── eda_petfinder.ipynb          # EDA ejecutado
│   └── eda_petfinder_dash_Roxy.py   # Dashboard Plotly Dash
├── 04_Tabulares_Optimizado_Roxy.ipynb   # FE v1/v2/v3 + Optuna
├── 05_Tabulares_CV_Roxy.ipynb           # FE v4/v5 + 5-fold CV
├── 06_Blend_BERT_LightGBM_Roxy.ipynb    # Transfer Learning blend
├── 07_BorutaSHAP_Roxy.ipynb             # Selección de features
├── 08_Ensemble_LGB_XGB_Roxy.ipynb       # Ensemble + Optuna XGB
├── 09_Blend_Triple_LGB_XGB_BERT_Roxy.ipynb  # Blend triple
└── 10_CatBoost_Ensemble_Roxy.ipynb      # CatBoost + ensemble final
```
