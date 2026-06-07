import sys

tex_content = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds, calc}

\geometry{a4paper, margin=2.5cm}

\title{Formulación MDP del problema de portafolio con costos de transacción}
\author{}
\date{}

\begin{document}
\maketitle

Consideremos un conjunto finito de activos $I$, un horizonte discreto $t = 1, \dots, T$, y un capital inicial $x_1 > 0$. El problema de portafolio con costos de transacción proporcional puede modelarse como un Proceso de Decisión de Markov (MDP) en tiempo discreto.

\section{Definición del MDP}
Definimos el MDP como $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma)$, donde:

\paragraph{Espacio de estados $\mathcal{S}$.} El estado al inicio del periodo $t$ se define como
$$s_t = (x_t, w_{t-1}, t),$$
donde:
\begin{itemize}
    \item $x_t \in \mathbb{R}^+$ es el capital disponible al inicio de $t$.
    \item $w_{t-1} = (w_{i,t-1})_{i \in I}$ son los pesos vigentes del portafolio al cierre de $t - 1$, con $w_{i,t-1} \geq 0$, $\sum_{i \in I} w_{i,t-1} = 1$.
    \item $t$ es el índice de tiempo (opcionalmente normalizado $t/T$ para efectos numéricos).
\end{itemize}
Por lo tanto, el espacio de estados es
$$\mathcal{S} = \left\{(x, w, t) : x \geq 0, w \in \Delta^{|I|-1}, t \in \{1, \dots, T\}\right\},$$
donde $\Delta^{|I|-1}$ denota el simplex de dimensión $|I| - 1$.

\paragraph{Espacio de acciones $\mathcal{A}$.} La acción en el periodo $t$ es la elección de un nuevo vector de pesos $w_t = (w_{i,t})_{i \in I}$:
$$a_t = w_t \in \mathcal{A},$$
sujeto a $w_{i,t} \geq 0$, $\sum_{i \in I} w_{i,t} = 1$.
En términos de RL continuo, $\mathcal{A}$ es también un simplex; en la implementación se trabaja con una parametrización continua que se proyecta al simplex.

\paragraph{Proceso de retornos.} Sea $R_t = (R_{i,t})_{i \in I}$ el vector de retornos en el intervalo $(t, t + 1]$. Supondremos que
$$R_t \sim D_t(\mu_t, \Sigma_t),$$
donde:
$$\mu_t = (\mu_{i,t})_{i \in I},$$
$$\Sigma_t = [\sigma_{ij,t}]_{i,j \in I}.$$
La distribución $D_t$ puede ser, por ejemplo, una distribución gaussiana multivariada o una mezcla condicionada a regímenes (bear/bull). Lo importante es que, dado $t$, la distribución de $R_t$ es conocida o estimada a partir de datos históricos.

\paragraph{Costos de transacción.} Dado el estado $(x_t, w_{t-1}, t)$ y la acción $w_t$, definimos el cambio de pesos
$$\Delta w_{i,t} = w_{i,t} - w_{i,t-1},$$
y la rotación total (en pesos del portafolio) como
$$\text{turn}_t = \sum_{i \in I} |\Delta w_{i,t}|.$$
Sea $c_i \geq 0$ la tasa de costo de transacción por unidad de rotación del activo $i$. El costo monetario en el periodo $t$ es
$$C_t(x_t, w_{t-1}, w_t) = x_t \sum_{i \in I} c_i |\Delta w_{i,t}|.$$

\paragraph{Dinámica de transición $\mathcal{P}$.} Dado el estado actual $s_t = (x_t, w_{t-1}, t)$, la acción $a_t = w_t$ y el retorno aleatorio $R_t$, definimos:
\begin{enumerate}
    \item Retorno del portafolio en el periodo $t$:
    $$r^{port}_t = \sum_{i \in I} w_{i,t} R_{i,t} = w_t^\top R_t.$$
    \item Capital al inicio de $t + 1$:
    $$x_{t+1} = x_t \left(1 + w_t^\top R_t - \sum_{i \in I} c_i |\Delta w_{i,t}|\right).$$
    \item Pesos para el siguiente estado: $w_t$ pasa a ser la componente de pesos del siguiente estado.
    \item Índice temporal: $t \to t + 1$.
\end{enumerate}
Por tanto, el kernel de transición queda implícitamente definido por:
$$S_{t+1} = (x_{t+1}, w_t, t + 1) = F(S_t, A_t, R_t),$$
con $R_t \sim D_t(\mu_t, \Sigma_t)$.

\paragraph{Función de recompensa $r(s, a)$.} Existen dos enfoques naturales y compatibles con el modelo de media--varianza con costos:
\begin{enumerate}
    \item Reward tipo media--varianza (determinístico):
    $$r_t(s_t, a_t) = \sum_{i \in I} w_{i,t} \mu_{i,t} - \lambda \sum_{i,j \in I} w_{i,t} w_{j,t} \sigma_{ij,t} - \sum_{i \in I} c_i |\Delta w_{i,t}|,$$
    donde $\lambda > 0$ es el parámetro de aversión al riesgo. Con $\gamma = 1$ y horizonte finito, el funcional $\sum_{t=1}^T r_t(s_t, a_t)$ reproduce la función objetivo de media--varianza con costos.
    \item Reward monetario (realizado):
    $$\tilde{r}_t(s_t, a_t, R_t) = x_{t+1} - x_t = x_t \left( w_t^\top R_t - \sum_{i \in I} c_i |\Delta w_{i,t}| \right),$$
    y el agente observa realizaciones de $\tilde{r}_t$ a partir de los retornos históricos o simulados.
\end{enumerate}
En ambos casos, el objetivo del agente es maximizar la suma descontada de recompensas:
$$J(\pi) = \mathbb{E}_\pi \left[ \sum_{t=1}^T \gamma^{t-1} r_t(S_t, A_t) \right],$$
para una política $\pi$.

\subsection*{1.2 Políticas y funciones de valor}
Una política estocástica se define como
$$\pi(w | x, w_{t-1}, t) = \Pr\{A_t = w | X_t = x, W_{t-1} = w_{t-1}, t\},$$
mientras que una política determinista se modela como
$$\mu : \mathcal{S} \to \mathcal{A}, \quad A_t = \mu(S_t).$$
Dada una política $\pi$, la función valor de estado es:
$$V^\pi(x, w_{t-1}, t) = \mathbb{E}_\pi \left[ \sum_{\tau=t}^T \gamma^{\tau-t} r_\tau(S_\tau, A_\tau) \Big| S_t = (x, w_{t-1}, t) \right].$$
La ecuación de Bellman para $V^\pi$ (con horizonte finito y $t < T$) es:
$$V^\pi(s_t) = \mathbb{E}_{A_t \sim \pi} \left[ r_t(s_t, A_t) + \gamma \mathbb{E}_{R_t} [V^\pi(S_{t+1})] \right].$$
Análogamente, la función valor acción--estado $Q^\pi$ es:
$$Q^\pi(s_t, a_t) = \mathbb{E} \left[ \sum_{\tau=t}^T \gamma^{\tau-t} r_\tau(S_\tau, A_\tau) \Big| S_t = s_t, A_t = a_t \right],$$
y satisface la ecuación de Bellman correspondiente.
Las funciones óptimas se definen por
$$V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s, a) = \max_\pi Q^\pi(s, a),$$
y una política óptima $\pi^*$ cumple
$$\pi^*(s) \in \arg\max_a Q^*(s, a).$$


\section{Metodología de implementación DRL (PPO y SAC)}
A continuación se presenta una metodología paso a paso para implementar este MDP de portafolio con costos de transacción usando aprendizaje por refuerzo profundo (DRL) con stable baselines3, PyTorch y los algoritmos PPO y SAC. La metodología se organiza en etapas coherentes desde la construcción del entorno hasta la comparación con la solución de referencia obtenida vía GAMS en un esquema de rolling horizon.

\subsection{Etapa 1: Preparación de datos y estimación de parámetros}
\begin{enumerate}
    \item \textbf{Datos de retornos.} Reunir series históricas semanales de retornos $R_{i,t}$ para todos los activos $i \in I$ en el horizonte de entrenamiento.
    \item \textbf{Estimación de momentos por periodo.} Para cada periodo $t$, estimar $\mu_t = \mathbb{E}[R_t]$, $\Sigma_t = \text{Cov}(R_t)$.
    Estos objetos $\mu_t$ y $\Sigma_t$ son equivalentes a los parámetros $\mu_{mix}$ y $\sigma_{mix}$ utilizados en el modelo GAMS.
    \item \textbf{Partición train/validation/test.} Dividir la serie de tiempo total en tres segmentos ordenados: Train, Validation, Test.
\end{enumerate}

\subsection{Etapa 2: Definición del entorno MDP (Gym-like)}
Implementar un entorno en la interfaz tipo Gym, por ejemplo \texttt{PortfolioEnv}, con los métodos \texttt{reset()} y \texttt{step(action)}:
\begin{enumerate}
    \item \textbf{Observaciones.} Definir el vector de observación en $t$ como
    $$o_t = \left[ x_t, w_{t-1}, t/T, \text{features}_t \right],$$
    donde $\text{features}_t$ se define explícitamente como los parámetros normalizados del modelo GAMS:
    $$\text{features}_t = \left[ \text{norm\_}\mu_{\text{mix}}(i,t), \text{norm\_var\_mix}(i,t), \text{norm\_cov\_mix}(t) \right].$$
    \item \textbf{Acciones.} El agente devuelve un vector continuo $\tilde{a}_t \in \mathbb{R}^{|I|}$ que se transforma a pesos factibles:
    $$w_{i,t} = \frac{\max(0, \tilde{a}_{i,t})}{\sum_{j \in I} \max(0, \tilde{a}_{j,t})}.$$
    \item \textbf{Step.} Dado el estado interno y la acción, el entorno evoluciona calculando costos, capital y reward.
    \item \textbf{Reset.} En \texttt{reset()}, inicializar capital, pesos, $t=1$.
\end{enumerate}

\vspace{1cm}
\begin{center}
\textbf{\Large Diagrama de Arquitectura PPO}
\end{center}
\vspace{0.5cm}

% -------------------------
% DIAGRAMA PPO TIKZ AQUI
% -------------------------
\begin{center}
\resizebox{\textwidth}{!}{
\begin{tikzpicture}[
    >=Latex,
    node distance=1.5cm and 2cm,
    font=\sffamily,
    % Estilos de cajas
    agentbox/.style={rectangle, rounded corners, draw=blue!60, fill=blue!5, very thick, align=center, inner sep=10pt},
    envbox/.style={rectangle, rounded corners, draw=green!60!black, fill=green!5, very thick, align=center, inner sep=10pt},
    mathbox/.style={rectangle, draw=gray!50, fill=white, align=center, inner sep=6pt},
    arrowlabel/.style={fill=white, text=black, font=\footnotesize, align=center}
]

% AGENTE PPO (Izquierda)
\node[agentbox] (actor) {
    \textbf{Red Actor (Política $\pi_\theta$)}\\[1ex]
    Input: $O_t$\\[1ex]
    Salida: $\mu_\theta, \sigma_\theta$
};

\node[mathbox, below=0.8cm of actor] (muestreo) {
    \textbf{Muestreo Estocástico}\\[1ex]
    $a_t \sim \mathcal{N}(\mu_\theta, \sigma_\theta)$
};

\node[agentbox, below=1.5cm of muestreo] (critic) {
    \textbf{Red Crítico (Valor $V_\phi$)}\\[1ex]
    Input: $O_t$\\[1ex]
    Salida: $V(O_t)$
};

\node[mathbox, below=0.8cm of critic] (ventaja) {
    \textbf{Cálculo de Ventaja ($A_t$)}\\[1ex]
    $A_t = r_t + \gamma V(O_{t+1}) - V(O_t)$
};

\node[mathbox, below=0.8cm of ventaja] (loss) {
    \textbf{Optimizador PPO}\\[1ex]
    $L^{CLIP} = \min(\text{ratio} \cdot A_t, \text{clip} \cdot A_t)$
};

% Contenedor del Agente
\begin{scope}[on background layer]
    \node[rectangle, rounded corners, draw=blue!80, fill=blue!2, dashed, very thick, fit=(actor) (muestreo) (critic) (ventaja) (loss), inner sep=15pt, label={[font=\large\bfseries, text=blue!80]above:Cerebro PPO (Agente On-Policy)}] (agent_bg) {};
\end{scope}

% Conexiones internas Agente
\draw[->, thick, blue] (actor) -- (muestreo);
\draw[->, thick, blue] (critic) -- (ventaja) node[midway, right, font=\footnotesize] {Predicción};
\draw[->, thick, blue, dashed] (ventaja) -- (loss) node[midway, right, font=\footnotesize] {Optimiza pesos};

% ENTORNO (Derecha)
\node[envbox, right=5cm of muestreo] (proyeccion) {
    \textbf{Proyección Símplex}\\[1ex]
    $w_{i,t} = \frac{\max(0, a_{i,t})}{\sum_j \max(0, a_{j,t})}$
};

\node[envbox, below=0.8cm of proyeccion] (dinamica) {
    \textbf{Evolución Capital}\\[1ex]
    $x_{t+1} = x_t \left(1 + \sum w_{i,t} R_{i,t} - c_i\right)$
};

\node[envbox, below=0.8cm of dinamica] (recompensa) {
    \textbf{Reward Media-Varianza}\\[1ex]
    $r_t = \sum w_{i,t} \mu_{i,t} - \lambda \sum\sum w_{i,t} w_{j,t} \sigma_{ij,t} - c_i$
};

% Contenedor del Entorno
\begin{scope}[on background layer]
    \node[rectangle, rounded corners, draw=green!60!black, fill=green!2, dashed, very thick, fit=(proyeccion) (dinamica) (recompensa), inner sep=15pt, label={[font=\large\bfseries, text=green!60!black]above:PortfolioEnv (Mercado)}] (env_bg) {};
\end{scope}

% Conexiones internas Entorno
\draw[->, thick, green!60!black] (proyeccion) -- (dinamica);
\draw[->, thick, green!60!black] (dinamica) -- (recompensa);

% INTERACCIONES (Ciclo RL)
\draw[->, line width=1.5pt, orange] (muestreo) -- (proyeccion) node[midway, arrowlabel] {Acción Continua\\$a_t \in \mathbb{R}^{|I|}$};

\draw[->, line width=1.5pt, red] (recompensa.west) to[out=180, in=0] node[pos=0.3, arrowlabel] {Señal $r_t$} (ventaja.east);

\coordinate (obs_out) at ($(recompensa.south) + (0, -0.5)$);
\draw[line width=1.5pt, purple] (recompensa.south) -- (obs_out);
\draw[line width=1.5pt, purple] (obs_out) -| ($(agent_bg.west) + (-1.5, 0)$) |- (actor.west) 
    node[pos=0.10, arrowlabel, xshift=2.5cm] {\textbf{Vector de Observación} $O_{t+1}$\\$O_{t+1} = \left[x_{t+1}, w_t, \frac{t}{T}, \text{norm\_}\mu_{\text{mix}}, \text{norm\_var}_{\text{mix}}, \text{norm\_cov}_{\text{mix}}\right]$};

\draw[->, line width=1.5pt, purple] ($(agent_bg.west) + (-1.5, 0)$) |- (critic.west);

\end{tikzpicture}
}
\end{center}

\subsection{Etapa 3: Definición de agentes y arquitecturas (PPO y SAC)}
\textbf{Espacio de observación y acción.} Configurar en stable baselines3:
\begin{itemize}
    \item Box continuo para el espacio de observación, con límites apropiados.
    \item Box continuo para el espacio de acción, con límites en $[-1, 1]$ y posterior proyección al simplex.
\end{itemize}

\textbf{PPO (on-policy, policy gradient).}
\begin{itemize}
    \item Usar PPO con política tipo MlpPolicy.
    \item Arquitectura sugerida: 2–3 capas densas (64–128 neuronas).
\end{itemize}

\textbf{SAC (off-policy, actor–critic maximum-entropy).}
\begin{itemize}
    \item Usar SAC con política MlpPolicy y acción continua.
    \item Arquitecturas similares para actor y críticos (2–3 capas densas).
\end{itemize}

\textbf{Capacidad de los agentes y error de generalización.}
Para minimizar el error de generalización:
\begin{enumerate}
    \item \textbf{Control de capacidad.} Ajustar la profundidad de la red para evitar sobreajuste evidente.
    \item \textbf{Regularización implícita.} En PPO, controlar clip range. En SAC, ajustar la temperatura de entropía.
    \item \textbf{Evaluación en validación.} Durante el entrenamiento, usar un evaluation callback que seleccione el modelo con mejor desempeño.
    \item \textbf{Walk-forward y múltiples semillas.} Entrenar con diferentes semillas y sub-períodos para medir robustez.
\end{enumerate}

\subsection{Etapa 4: Diseño de pruebas de comparación}
\begin{enumerate}
    \item \textbf{Base determinista (GAMS, media–varianza con costos).}
    Resolver el modelo en ventanas móviles de longitud $H$, aplicando la primera decisión y avanzando al siguiente periodo $t+1$.
    \item \textbf{Agentes DRL (PPO y SAC).}
    Evaluar en el periodo de test usando retornos realizados y el mismo esquema de costos.
    \item \textbf{Benchmarks adicionales.}
    Portafolio buy-and-hold (50/50) y rebalanceo periódico fijo.
\end{enumerate}

\subsection{Etapa 5: Métricas de evaluación y comparación}
\begin{itemize}
    \item Retorno acumulado: $\text{RetAcum} = x_T/x_1 - 1$.
    \item Volatilidad del portafolio.
    \item Ratio de Sharpe, semivarianza, drawdowns máximos.
    \item Rotación y costos: promedio de $\text{turn}_t$ y costo acumulado.
    \item Error de generalización: $\text{Regret}_{\text{GAMS}}^{\text{DRL}} = \text{RetAcum}_{\text{GAMS}} - \text{RetAcum}_{\text{DRL}}$.
\end{itemize}

\subsection{Etapa 6: Síntesis}
La metodología propuesta permite:
\begin{enumerate}
    \item Formular el problema de portafolio con costos de transacción como un MDP bien definido.
    \item Entrenar agentes DRL (PPO y SAC) que aprenden políticas a partir de datos.
    \item Controlar la capacidad de los agentes y el error de generalización mediante particiones temporales.
    \item Comparar de forma rigurosa el desempeño DRL contra GAMS en un esquema rolling horizon.
\end{enumerate}

\end{document}
"""

with open("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex", "w", encoding="utf-8") as f:
    f.write(tex_content)
print("done")
