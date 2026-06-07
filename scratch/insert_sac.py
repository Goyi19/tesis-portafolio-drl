import sys

with open("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

insert_idx = -1
for i, line in enumerate(lines):
    if "\\subsection{Etapa 4: Diseño de pruebas de comparación}" in line:
        insert_idx = i
        break

if insert_idx != -1:
    sac_content = r"""
\vspace{0.5cm}
\textbf{Definiciones para Arquitectura SAC.} 
Para el agente \textbf{SAC (Soft Actor-Critic)}, el objetivo se expande para maximizar tanto el retorno esperado como la entropía de la política. El proceso incluye componentes específicos para su naturaleza off-policy:
\begin{itemize}
    \item \textbf{Replay Buffer ($\mathcal{D}$):} Memoria donde se almacenan las transiciones observadas $(O_t, a_t, r_t, O_{t+1})$. Permite reutilizar experiencia pasada.
    \item \textbf{Doble Crítico Soft ($Q_{\phi_1}, Q_{\phi_2}$):} Dos redes neuronales que estiman el valor acción-estado continuo $Q(O_t, a_t)$. Se utiliza el mínimo de ambas estimaciones para evitar el sesgo de sobreestimación del valor.
    \item \textbf{Optimizador SAC ($\alpha$, $\tau$):} Incluye el cálculo y optimización de las \textbf{Redes Target} mediante una media móvil exponencial con tasa $\tau$, y la gestión del \textbf{Coeficiente de Entropía ($\alpha$)}, el cual controla la temperatura de exploración.
\end{itemize}

A continuación se presenta la arquitectura conceptual para el agente SAC, la cual interactúa con el mismo entorno definido previamente:

\vspace{0.5cm}
\begin{center}
\textbf{\Large Diagrama de Arquitectura SAC}
\end{center}
\vspace{0.3cm}

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

% AGENTE SAC (Izquierda)
\node[agentbox] (actor) {
    \textbf{Red Actor (Política $\pi_\theta$)}\\[1ex]
    Input: $O_t$\\[1ex]
    Salida: $\mu_\theta, \sigma_\theta$
};

\node[mathbox, below=0.8cm of actor] (muestreo) {
    \textbf{Muestreo Estocástico ($a_t$)}
};

\node[mathbox, below=1cm of muestreo] (buffer) {
    \textbf{Replay Buffer ($\mathcal{D}$)}
};

\node[agentbox, below=1cm of buffer] (critic) {
    \textbf{Doble Crítico Soft ($Q_{\phi_1}, Q_{\phi_2}$)}\\[1ex]
    Input: $O_t, a_t$\\[1ex]
    Salida: $Q(O_t, a_t)$
};

\node[mathbox, below=0.8cm of critic] (loss) {
    \textbf{Optimizador SAC ($\alpha$, $\tau$)}
};

% Contenedor del Agente
\begin{scope}[on background layer]
    \node[rectangle, rounded corners, draw=blue!80, fill=blue!2, dashed, very thick, fit=(actor) (muestreo) (buffer) (critic) (loss), inner sep=15pt, label={[font=\large\bfseries, text=blue!80]above:Cerebro SAC (Agente Off-Policy)}] (agent_bg) {};
\end{scope}

% Conexiones internas Agente
\draw[->, thick, blue] (actor) -- (muestreo);
\draw[->, thick, blue, dashed] (buffer) -- (critic) node[midway, right, font=\footnotesize] {Minibatch};
\draw[->, thick, blue, dashed] (critic) -- (loss) node[midway, right, font=\footnotesize] {Optimiza pesos};

% ENTORNO (Derecha) - Alineado con SAC
\node[envbox, right=6cm of muestreo] (proyeccion) {
    \textbf{Proyección Símplex ($w_t$)}
};

\node[envbox, below=1.2cm of proyeccion] (dinamica) {
    \textbf{Evolución Capital ($x_{t+1}$)}
};

\node[envbox, below=1.2cm of dinamica] (recompensa) {
    \textbf{Reward Media-Varianza ($r_t$)}
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

% La señal de recompensa va al buffer
\draw[->, line width=1.5pt, red] (recompensa.west) -- (buffer.east) node[midway, fill=white, font=\footnotesize, align=center] {Transición\\$(O_t, a_t, r_t, O_{t+1})$};

\coordinate (obs_out) at ($(recompensa.south) + (0, -0.5)$);
\draw[line width=1.5pt, purple] (recompensa.south) -- (obs_out);
\draw[line width=1.5pt, purple] (obs_out) -| ($(agent_bg.west) + (-1.5, 0)$) |- (actor.west) 
    node[pos=0.10, arrowlabel, xshift=3.5cm] {\textbf{Vector de Observación} $O_{t+1}$\\$O_{t+1} = \left[x_{t+1}, w_t, \frac{t}{T}, \text{norm\_}\mu_{\text{mix}}, \text{norm\_var}_{\text{mix}}, \text{norm\_cov}_{\text{mix}}\right]$};

\draw[->, line width=1.5pt, purple] ($(agent_bg.west) + (-1.5, 0)$) |- (critic.west);

\end{tikzpicture}
}
\end{center}
\vspace{1cm}

"""
    lines.insert(insert_idx, sac_content)
    
    with open("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("inserted sac")
else:
    print("could not find insertion point")
