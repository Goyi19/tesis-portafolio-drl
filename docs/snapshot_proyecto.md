# Snapshot de Contexto: Proyecto Tesis DRL Portafolios

Este documento resume el estado actual del pensamiento y diseño del proyecto al finalizar la fase teórica inicial. **Nota:** No constituye una definición sólida; es material de referencia para ser cuestionado y refinado en la fase de implementación.

## 1. El Pilar GAMS (Clase 0)
*   **Origen:** Archivo `ps.gms` (HMM + Markowitz).
*   **Hallazgo Crítico:** La estimación de $\mu$ (media) y $\sigma$ (covarianza) en GAMS es global ($t=1 \ldots 163$). 
*   **Pendiente de Implementación:** Para evitar el *data leakage* detectado por el profesor, el $\mu$ y $\sigma$ pasados al DRL deben ser calculados en "ventana expansiva" (rolling), usando solo información del pasado en cada paso $t$.

## 2. Marco Conceptual del MDP (En Revisión)
*   **Estado ($S_t$):** Actualmente pensado como un vector que concatena $[\mu_{rolling}, \sigma_{rolling}, P(\text{bear})_t, w_{t-1}]$. Sujeto a cambios de dimensionalidad.
*   **Acción ($a_t$):** Pesos del portafolio ($w$), usualmente proyectados al Símplex ($\sum w_i = 1, w_i \ge 0$).
*   **Recompensa ($r_t$):** Basada en la función objetivo de GAMS (Retorno - Riesgo - Costos), pero calculada paso a paso en el entorno `Gymnasium`.

## 3. Arquitectura y Modelos (Referencia SB3)
*   **Modelos Base:** PPO (Proximal Policy Optimization) y SAC (Soft Actor-Critic).
*   **Implementación:** Se contempla el uso de `Stable-Baselines3` como punto de partida, pero la arquitectura de las redes (MLP, capas, activaciones) y el manejo de los tensores debe ser validado empíricamente.
*   **Matemática de Aprendizaje:** Se han discutido las fórmulas de Ventaja ($A_t$), TD-Error y el optimizador ADAM. Estos desarrollos servirán como guía lógica para depurar el código cuando los resultados no coincidan con la teoría.

## 4. Próxima Tarea: Implementación Fase 1
1.  **Data Handler:** Desarrollar el script de Python que replique la lógica de `ps.gms` pero con el rigor de la ventana expansiva para los momentos estadísticos.
2.  **PortfolioEnv:** Construir el entorno en `Gymnasium` que procese estos datos.
3.  **Refinamiento PPT:** Mejorar el material visual para el profesor integrando estas correcciones de "agencia" y rigor matemático.

---
*Este documento marca el fin de la Fase 0 (Diseño) y el inicio de la Fase 1 (Código).*
