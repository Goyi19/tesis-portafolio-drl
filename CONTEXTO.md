# Contexto del Proyecto: Tesis Portafolios DRL

## 1. Información General
* **Autor:** Rodrigo Valenzuela T.
* **Profesor Guía:** Juan Pérez
* **Tema:** Optimización de construcción de portafolios de inversión con Deep Reinforcement Learning (DRL).
* **Objetivo Principal:** Aprender una política de asignación de portafolio que determine las proporciones a invertir en cada activo maximizando el retorno esperado, ajustado por riesgo y costos de transacción, considerando regímenes de mercado (Bear/Bull) modelados con HMM (Hidden Markov Models).

## 2. Estado Actual del Proyecto
* **Hito 1 (Completado):** Se estudiaron y definieron los modelos matemáticos base (Markowitz, Entropía de Shannon, Modelo Naive, Ratios de Sharpe y Sortino). Se introdujo la teoría base de Reinforcement Learning (MDP, Ecuación de Bellman).
* **Hito 2 (Completado):** Se formuló el problema como un POMDP continuo. Se definió la arquitectura DRL (PPO/SAC con redes recurrentes LSTM/RNN) para competir contra el optimizador determinista en GAMS.
* **Modelo Teórico y HMM (Completado):** Se documentó exhaustivamente la matemática detrás de GAMS, incluyendo la función objetivo con retornos mixtos ($\mu^{mix}$), covarianza ($\Sigma^{mix}$), aversión al riesgo ($\lambda$) y costos de transacción ($c_i$).

## 3. Decisiones Clave Tomadas
* El agente DRL debe emular el comportamiento del optimizador GAMS pero sin mirar el futuro completo, usando solo datos históricos pasados (Frame Stacking) y aproximando la probabilidad de estados Bear/Bull.
* Se incluirá una penalización por Entropía en el DRL para forzar la exploración inicial y evitar que el modelo se congele en un solo activo.

## 4. Próximos Pasos Inmediatos
* **Paso Actual:** Codificar estrictamente el bloque "MOTOR" en Python usando **IPOPT (Pyomo)**.
* **Meta:** Lograr que al pasarle a Python los mismos datos crudos del profesor, el script arroje el **mismo rendimiento numérico exacto** que el modelo GAMS. Esta será la prueba de validación antes de entrenar al agente DRL.

---
*Nota para la IA: Si estás leyendo esto al inicio de una sesión en un nuevo computador, utiliza este documento para recuperar el contexto y continuar trabajando en la sección de "Próximos Pasos Inmediatos".*
