# PC4 — Predicción de Series Temporales con Redes Recurrentes (Rossmann)

## 📌 Descripción del Proyecto

Proyecto de consolidación del Máster en Inteligencia Artificial (MBIT) enfocado en la **predicción de ventas diarias** de tiendas Rossmann usando **redes recurrentes profundas (LSTM)**.

**Objetivo:** Construir un modelo que maximice el R² en test, sin usar la variable `Customers` (información futura).

---

## 📊 Resultados Finales

### Modelos LSTM (notebooks Colab)

| Tienda | Modelo | R² Train | R² Test | RMSE Test | Lookback |
|--------|--------|----------|---------|-----------|----------|
| **Tienda 1** | LSTM + OHE | 93.7% | **90.4%** ✅ | 718 | 7 días |
| **Tienda 2** | LSTM + OHE | 86.7% | **80.0%** | 1027 | 28 días |
| Tienda 1 | LSTM + Embeddings | 96.0% | 49.8% | — | 14 días |
| Tienda 2 | LSTM + Embeddings | 85.1% | 71.4% | — | 7 días |

> **Nota:** El enfoque con **One-Hot Encoding (OHE)** supera significativamente a los **Embeddings** en este dataset.

### Comparativa con Baselines (Tienda 1)

| Modelo | MAE (test) | RMSE (test) | R² (test) |
|--------|------------|-------------|-----------|
| Naïve (t-1) | 2172 | 3097 | -0.76 |
| Rolling Mean 7d | 1868 | 2423 | -0.07 |
| Linear Regression | 748 | 954 | 0.83 |
| Random Forest | 388 | 633 | 0.93 |
| **LSTM (OHE)** | — | **718** | **0.90** |

---

## 📁 Estructura del Proyecto

```
PC4/
├── ejemplos/                          # Notebooks de referencia (SARIMAX, AutoARIMA, LSTMs)
├── PC_series_temporales/
│   ├── data/                          # Datasets (CSVs)
│   │   ├── datos_diarios_tienda1.csv
│   │   ├── datos_diarios_tienda2.csv
│   │   ├── df_tienda1_preprocesado.csv
│   │   └── ...
│   │
│   ├── notebook_colab/                # 🔥 NOTEBOOKS FINALES (Colab)
│   │   ├── PC4_Series_temporales_T1_LSTM_endógena_y_exógenas.ipynb
│   │   ├── PC4_Series_temporales_T2_LSTM_endógena_y_exógenas.ipynb
│   │   ├── PC4_series_temporales_T1_embeddings.ipynb
│   │   └── PC4_series_temporales_T2_embeddings.ipynb
│   │
│   ├── Store6_EDA_Visualizacion.ipynb          # 📈 EDA introductorio
│   ├── Store6_Feature_Eng&Baselines.ipynb      # 🔧 Feature Engineering + Baselines
│   ├── PC4-Baseline2_LSTM.ipynb                # 🧠 Baseline LSTM (local)
│   └── metrics_table.csv                       # Tabla de métricas baselines
│
└── README.md
```

---

## 📓 Descripción de Notebooks

### 1️⃣ EDA & Visualización (`Store6_EDA_Visualizacion.ipynb`)
**Entorno:** Local  
**Propósito:** Entender el dataset antes de modelar.

- Carga y validación de datos (881 registros, 2013-2015)
- Chequeos de calidad: duplicados, gaps temporales
- Descomposición STL, ACF/PACF, tests ADF/KPSS
- Análisis por día de semana y mes
- Detección de outliers y patrones estacionales

---

### 2️⃣ Feature Engineering & Baselines (`Store6_Feature_Eng&Baselines.ipynb`)
**Entorno:** Local  
**Propósito:** Crear features y establecer baselines de referencia.

**Baselines implementados:**
- **Naïve:** Pred(t) = Real(t-1)
- **Rolling Mean 7d:** Media móvil de los últimos 7 días
- **Linear Regression** y **Random Forest** como referencias ML

**Feature Engineering:**
- Variables de calendario: `year`, `month`, `day`, `dayofweek`, `is_weekend`
- Codificación cíclica: `month_sin/cos`, `dow_sin/cos`
- One-Hot Encoding para categóricas (Promo, StateHoliday, etc.)
- Verificación de incongruencias (Customers/Open/Sales)

**Conclusión:** Los baselines estadísticos tienen R² negativo → necesidad de modelos más avanzados.

---

### 3️⃣ Baseline LSTM (`PC4-Baseline2_LSTM.ipynb`)
**Entorno:** Local  
**Propósito:** Primer modelo LSTM para comparación.

**Configuración:**
| Parámetro | Valor |
|-----------|-------|
| TIME_STEPS (lookback) | 30 |
| LSTM_UNITS | 64 |
| DROPOUT | 0.15 |
| BATCH_SIZE | 64 |
| EPOCHS_MAX | 500 (EarlyStopping) |
| Split | 80% train / 20% test |

**Variables de entrada (multivariante):**
- Target: `Sales` (escalado con StandardScaler)
- Binarias: `Open`, `Promo`, `SchoolHoliday`, `StateHoliday_*`
- Cíclicas: `day_year_sin/cos`, `dow_sin/cos`, `day_month_sin/cos`, `year_norm`

**Métricas:** MAE, sMAPE, R²

---

### 4️⃣ Notebooks Finales — Colab (`notebook_colab/`)
**Entorno:** Google Colab  
**Propósito:** Modelos optimizados con mejores resultados.

#### a) LSTM + One-Hot Encoding (✅ Mejor resultado)
- `PC4_Series_temporales_T1_LSTM_endógena_y_exógenas.ipynb`
- `PC4_Series_temporales_T2_LSTM_endógena_y_exógenas.ipynb`

**Resultados Tienda 1:**
```
R2 del modelo en training      :  0.937
R2 del modelo en test          :  0.904
RMSE del modelo en test        :  718
```

**Resultados Tienda 2:**
```
R2 del modelo en training      :  0.867
R2 del modelo en test          :  0.800
RMSE del modelo en test        :  1027
```

**Enfoque:**
- One-Hot Encoding para `Month` (12 cols) y `DayOfWeek` (7 cols)
- Variables binarias: Open, Promo, SchoolHoliday, StateHoliday_*
- Enventanado: `lookback = 7` días (T1) y `lookback = 28` días (T2)
- Escalado: StandardScaler solo en target
- Arquitectura: LSTM(5 neuronas) → Dense(1)
- Optimizador: Adam

#### b) LSTM + Embeddings (Alternativa)
- `PC4_series_temporales_T1_embeddings.ipynb`
- `PC4_series_temporales_T2_embeddings.ipynb`

**Resultados Tienda 1:**
```
R2 en training: 0.960
R2 en val:      0.878
R2 en test:     0.498
```

**Resultados Tienda 2:**
```
R2 en training: 0.851
R2 en val:      0.753
R2 en test:     0.714
```

**Enfoque:**
- Capas Embedding para variables categóricas (month, day_week, day_month)
- `dim_embedding = 3`
- Permite representaciones densas aprendidas
- **Resultado inferior al OHE** (posible overfitting en embeddings)

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Keras / TensorFlow** — Redes LSTM
- **Scikit-learn** — Preprocesado, métricas, baselines ML
- **Statsmodels** — STL, ACF/PACF, ADF/KPSS
- **Pandas / NumPy** — Manipulación de datos
- **Matplotlib / Seaborn** — Visualización

---

## 📈 Datos

**Dataset:** Ventas diarias de tiendas Rossmann (Kaggle)  
**Período:** 2013-01-01 → 2015-05-31 (~881 registros por tienda)

| Variable | Descripción |
|----------|-------------|
| `Sales` | Ventas diarias (TARGET) |
| `Open` | Tienda abierta (0/1) |
| `Promo` | Promoción activa (0/1) |
| `SchoolHoliday` | Vacaciones escolares (0/1) |
| `StateHoliday_*` | Festivos estatales (one-hot) |
| `DayOfWeek` | Día de la semana (1-7) |
| `Customers` | ❌ No usar (información futura) |

---

## 🚀 Cómo Ejecutar

### Local (EDA, Feature Eng, Baseline2)
```bash
cd PC_series_temporales/
jupyter notebook Store6_EDA_Visualizacion.ipynb
```

### Google Colab (Notebooks finales)
1. Subir notebooks de `notebook_colab/` a Colab
2. Montar Google Drive cuando se solicite
3. Ejecutar celdas secuencialmente

---

## 📝 Conclusiones

1. Los **baselines estadísticos** (Naïve, Rolling Mean) son insuficientes (R² < 0)
2. **ML clásico** (Random Forest) alcanza R² = 0.93 en Tienda 1
3. **LSTM con One-Hot Encoding** logra R² = **0.90** en Tienda 1 y **0.80** en Tienda 2
4. **One-Hot Encoding supera a Embeddings** en este dataset (menos overfitting)
5. El enfoque **many-to-one** con ventana deslizante es efectivo para pronóstico de 1 día
6. La Tienda 2 presenta mayor dificultad de predicción (posiblemente por patrones más erráticos)

---

## 👤 Autor

**Francisco Alvarez**  
Adaptación de notebooks originales de Manuel Sánchez-Montañés

