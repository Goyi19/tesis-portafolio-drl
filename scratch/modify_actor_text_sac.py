import sys

def modify_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # We want to change the actor text ONLY in the SAC diagram.
    # The SAC diagram starts around "AGENTE SAC" or "Cerebro SAC".
    
    parts = content.split("AGENTE SAC (Izquierda)")
    if len(parts) > 1:
        parts[1] = parts[1].replace(
            r"\textbf{Red Actor (Política $\pi_\theta$)}",
            r"\textbf{Red Actor (Política Determinista $\mu_\theta$)}"
        )
        content = "AGENTE SAC (Izquierda)".join(parts)
    else:
        # For standalone diagram
        content = content.replace(
            r"\textbf{Red Actor (Política $\pi_\theta$)}",
            r"\textbf{Red Actor (Política Determinista $\mu_\theta$)}"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

modify_file("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex")
modify_file("docs/Registro de Definiciones del Proyecto/diagrama_sac_tikz.tex")
print("done_changing_actor_sac")
