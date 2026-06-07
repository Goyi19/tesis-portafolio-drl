import sys

def modify_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Modificar Actor
    content = content.replace(
r"""\node[agentbox] (actor) {
    \textbf{Red Actor (Política $\pi_\theta$)}\\[1ex]
    Input: $O_t$\\[1ex]
    Salida: $\mu_\theta, \sigma_\theta$
};""",
r"""\node[agentbox] (actor) {
    \textbf{Red Actor (Política $\pi_\theta$)}\\[1ex]
    Input: $O_t$\\[1ex]
    Salida: $a_t$ (Acción Continua)
};"""
    )
    
    # Eliminar muestreo
    content = content.replace(
r"""\node[mathbox, below=0.8cm of actor] (muestreo) {
    \textbf{Muestreo Estocástico ($a_t$)}
};""", ""
    )
    
    # Cambiar referencia de buffer
    content = content.replace(
r"""\node[mathbox, below=1cm of muestreo] (buffer) {""",
r"""\node[mathbox, below=1cm of actor] (buffer) {"""
    )
    
    # Eliminar muestreo de fit
    content = content.replace(
r"""fit=(actor) (muestreo) (buffer)""",
r"""fit=(actor) (buffer)"""
    )
    
    # Eliminar conexion actor -> muestreo
    content = content.replace(
r"""\draw[->, thick, blue] (actor) -- (muestreo);""", ""
    )
    
    # Cambiar referencia de proyeccion
    content = content.replace(
r"""\node[envbox, right=6cm of muestreo] (proyeccion) {""",
r"""\node[envbox, right=6cm of actor, yshift=-1.5cm] (proyeccion) {"""
    )
    
    # Cambiar conexion muestreo -> proyeccion a actor -> proyeccion
    content = content.replace(
r"""\draw[->, line width=1.5pt, orange] (muestreo) -- (proyeccion)""",
r"""\draw[->, line width=1.5pt, orange] (actor) -- (proyeccion)"""
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

modify_file("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex")
modify_file("docs/Registro de Definiciones del Proyecto/diagrama_sac_tikz.tex")
print("done_modifying_sac")
