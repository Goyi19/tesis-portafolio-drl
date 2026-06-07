# Contexto del Proyecto: Tesis Portafolios DRL

## 1. Información General
* **Autor:** Rodrigo Valenzuela T.
* **Profesor Guía:** Juan Pérez
* **Tema:** Optimización de construcción de portafolios de inversión con Deep Reinforcement Learning (DRL).
* **Objetivo Principal:** Desarrollar y comparar políticas de asignación de portafolio mediante algoritmos PPO y SAC que maximicen el retorno ajustado por riesgo y costos de transacción, frente a modelos deterministas clásicos (GAMS) y estrategias Buy-and-Hold.

## 2. Estado Actual del Proyecto
* **Fase de Desarrollo y Código (Completada):** Ya se ha implementado con éxito todo el entorno computacional en Python y se han corrido los experimentos.
  * Entorno Gym (`portfolio_env.py`) creado.
  * Agentes DRL (`ppo_agent.py` y `sac_agent.py`) entrenados e implementados.
  * Análisis de sensibilidad (aversión al riesgo $\lambda$ y costos de transacción $c_i$) ejecutados. Todos los gráficos de métricas y validación están generados en la carpeta `results/figures/`.
* **Fase de Escritura de la Memoria (Actual):** Nos encontramos redactando la memoria formal de título en base al documento "Estructura Capitulos Memoria". El documento abarca 6 capítulos principales (Introducción, Marco Teórico, Metodología, Implementación y Resultados, Sensibilidad y Conclusiones).

## 3. Decisiones Clave y Metodología
* **Datos:** Se usan activos S&P 500 (SPX) y Criptomonedas (CMC200), particionados en Train (70%), Val (15%) y Test (15%).
* **Arquitectura:** Se utilizaron modelos PPO y SAC con arquitecturas recurrentes (LSTM/RNN) para lidiar con la observabilidad parcial del mercado (regímenes Bear/Bull, deducidos con HMM).
* **Rúbrica de Redacción (Crítico):** Toda la escritura debe seguir formato estrictamente académico: tercera persona (voz impersonal), estilo formal, párrafos con una sola idea principal, uso de conectores discursivos, y estilo de citas Harvard.

## 4. Próximos Pasos Inmediatos
* **Pendiente para Cierre de Resultados:** Integrar los valores del benchmark teórico GAMS (los cuales se generarán externamente por problemas de licencia) en los gráficos actuales. Esto permitirá tener la curva de GAMS comparada lado a lado con los 20 modelos entrenados (PPO, SAC) y los benchmarks actuales (Naive B&H).
* **Paso Actual:** Iniciar la redacción de los capítulos de la tesis en simultáneo, de acuerdo a la "Rúbrica de Escritura", sin necesidad de esperar la integración final de la curva GAMS.
* *(Nota: Actualizar este punto al finalizar cada sesión de trabajo para detallar exactamente en qué capítulo, párrafo o sección de la memoria nos quedamos trabajando)*.

---
*Nota para la IA: Si estás leyendo esto al inicio de una sesión en un nuevo computador, utiliza este documento para recuperar el contexto y continuar trabajando en la sección de "Próximos Pasos Inmediatos".*
