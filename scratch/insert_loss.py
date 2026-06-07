import sys

def modify_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    insert_target = r"""\textbf{Capacidad de los agentes y error de generalización.}"""
    
    loss_text = r"""
\textbf{Funciones de Pérdida (Loss) para Backpropagation.}
Para materializar el aprendizaje, los algoritmos optimizan las siguientes funciones de pérdida mediante gradiente descendente:

\textbf{Loss de PPO:}
El agente PPO actualiza su política $\pi_\theta$ maximizando una función objetivo recortada (Clipped Surrogate Objective) para evitar actualizaciones destructivas:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
donde $\rho_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ es la razón de probabilidad y $\epsilon$ es el rango de recorte. Simultáneamente, el Crítico $V_\phi$ se actualiza minimizando el error cuadrático medio respecto a los retornos calculados:
$$L^{VF}(\phi) = \hat{\mathbb{E}}_t \left[ \left( V_\phi(s_t) - V_t^{\text{target}} \right)^2 \right]$$

\textbf{Loss de SAC:}
En SAC, los parámetros de las redes críticas gemelas $Q_{\phi_i}$ se optimizan minimizando el Error Cuadrático Medio de la ecuación de Bellman \textit{Soft}:
$$L_Q(\phi_i) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}} \left[ \left( Q_{\phi_i}(s_t, a_t) - \left( r_t + \gamma \left( \min_{j=1,2} Q_{\phi_{\text{target}, j}}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi_\theta(\tilde{a}_{t+1} | s_{t+1}) \right) \right) \right)^2 \right]$$
Por su parte, el Actor $\pi_\theta$ actualiza sus pesos minimizando la pérdida de política basada en la divergencia Kullback-Leibler, lo que se traduce en:
$$L_\pi(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}, \tilde{a}_t \sim \pi_\theta} \left[ \alpha \log \pi_\theta(\tilde{a}_t | s_t) - \min_{j=1,2} Q_{\phi_j}(s_t, \tilde{a}_t) \right]$$

"""
    
    content = content.replace(insert_target, loss_text + insert_target)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

modify_file("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex")
print("inserted_loss_functions")
