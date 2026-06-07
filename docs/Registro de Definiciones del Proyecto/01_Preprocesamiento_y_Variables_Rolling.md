# 01. Preparación de Datos y Estimación de Parámetros (Etapa 1)

Este documento detalla la ejecución técnica de la **Etapa 1** definida en la Memoria de Referencia (Documento 02), adaptándola a los activos SPX y CMC200 para un horizonte de $T=163$ semanas.

## 1. Datos Fuente y Horizonte Temporal

El proyecto utiliza series históricas semanales para el periodo completo de $T=163$ pasos. Los insumos base son:
*   **Retornos Netos ($R_{i,t}$):** Observaciones semanales de SPX y CMC200.
*   **Probabilidades HMM ($P_{i,k,t}$):** Señales exógenas de régimen (Bear/Bull).

## 2. Partición del Dataset (Rigor DRL)

Siguiendo la metodología *walk-forward* de la Biblia, dividimos las 163 semanas para garantizar la capacidad de generalización:

| Conjunto | Rango Semanal (Aprox) | Propósito |
| :--- | :--- | :--- |
| **Entrenamiento (Train)** | 1 - 114 (70%) | Aprendizaje de la política del agente. |
| **Validación (Val)** | 115 - 138 (15%) | Ajuste de hiperparámetros y control de sobreajuste. |
| **Prueba (Test)** | 139 - 163 (15%) | Evaluación final *out-of-sample* (Comparación vs GAMS). |

## 3. Estimación de Parámetros Mix

Para cada periodo $t$, calculamos las estimaciones que conformarán las **Features** del agente. Este proceso se realiza de forma expansiva para eliminar el *data leakage*.

### 3.1 Estimación de Momentos por Régimen
Para cada activo $i$ bajo el régimen $k$, los momentos se estiman de forma expansiva ponderando cada observación histórica por su probabilidad de pertenencia al régimen:

**Media estimada:**
$$ \hat{\mu}_{i,k,t} = \frac{\sum_{m=1}^t P_{i,k,m} R_{i,m}}{\sum_{m=1}^t P_{i,k,m}} $$

**Covarianza estimada:**
$$ \hat{\sigma}_{i,j,k,t} = \frac{\sum_{m=1}^t P_{i,k,m} (R_{i,m} - \hat{\mu}_{i,k,t})(R_{j,m} - \hat{\mu}_{j,k,t})}{\sum_{m=1}^t P_{i,k,m}} $$

### 3.2 Construcción de Variables Mix
$$ \mu_{mix, i,t} = \sum_{k} P_{i,k,t} \cdot \hat{\mu}_{i,k,t} $$
$$ \sigma_{mix, i,j,t} = \sum_{k} P_{i,k,t} \cdot \hat{\sigma}_{i,j,k,t} $$

Estas variables integran la incertidumbre del régimen en señales únicas, simplificando la tarea del agente DRL al entregarle la información ya procesada.

## 4. Normalización Z-Score de Observaciones

Para cumplir con la estabilidad numérica exigida (Etapa 2.1 de la Biblia), aplicamos una normalización expansiva a los resultados del Mix. Para cada variable mix $x_{mix, t}$, calculamos su versión normalizada $z(x)_{mix, t}$ utilizando su propia media y desviación estándar acumulada hasta $t$:
$$ z(x)_{mix, t} = \frac{x_{mix, t} - \mu(x)_{mix, t}}{\sigma(x)_{mix, t}} $$

## 6. Parámetros del Modelo y Configuración Base (Referencia GAMS)

Para asegurar la comparabilidad entre el modelo DRL y el optimizador determinista, se registran los siguientes parámetros extraídos del modelo GAMS original:

*   **Capital Inicial ($x_0$):** 10,000 USD.
*   **Pesos Iniciales ($w_0$):** 50% SPX / 50% CMC200.
*   **Costos de Transacción ($c_i$):**
    *   SPX: 0.005 ($0.5\%$).
    *   CMC200: 0.010 ($1.0\%$).
*   **Aversión al Riesgo ($\lambda$):**
    *   Grilla GAMS: $\{0.05, 0.10, 0.25, 0.50, 1.00\}$.
    *   **Selección DRL:** Se utilizará $\lambda = 0.10$ como estándar para el entrenamiento y recompensa.

---
**Próximo Paso:** Proceder con la implementación del entorno Gymnasium siguiendo las directrices del **Documento 02 (Memoria de Referencia)**.
