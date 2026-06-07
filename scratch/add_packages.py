import sys

def modify_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    packages = r"""\usepackage{graphicx}
\usepackage{geometry}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tcolorbox}
"""
    content = content.replace(r"\usepackage{graphicx}" + "\n" + r"\usepackage{geometry}" + "\n" + r"\usepackage{tikz}", packages)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

modify_file("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex")
print("added_packages")
