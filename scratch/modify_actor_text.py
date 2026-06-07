import sys

def modify_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Cambiar texto de actor en el diagrama SAC (y PPO si lo encuentra, pero el de SAC seguro)
    # Solo reemplazamos la politica pi_theta por Determinista mu_theta
    content = content.replace(
        r"\textbf{Red Actor (Política $\pi_\theta$)}",
        r"\textbf{Red Actor (Política Determinista $\mu_\theta$)}"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

modify_file("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex")
modify_file("docs/Registro de Definiciones del Proyecto/diagrama_sac_tikz.tex")
modify_file("docs/Registro de Definiciones del Proyecto/diagrama_ppo_tikz.tex")
print("done_changing_actor")
