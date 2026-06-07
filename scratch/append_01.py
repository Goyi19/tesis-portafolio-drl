import sys

# Read 01
with open("docs/Registro de Definiciones del Proyecto/01_Preprocesamiento_y_Variables_Rolling.tex", "r", encoding="utf-8") as f:
    content_01 = f.read()

# Extract body
start_marker = r"\maketitle"
end_marker = r"\end{document}"

start_idx = content_01.find(start_marker) + len(start_marker)
end_idx = content_01.find(end_marker)

if start_idx > len(start_marker) and end_idx != -1:
    body_01 = content_01[start_idx:end_idx].strip()
    
    # Read 02
    with open("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex", "r", encoding="utf-8") as f:
        content_02 = f.read()
    
    end_02_idx = content_02.find(end_marker)
    if end_02_idx != -1:
        new_content_02 = content_02[:end_02_idx] + "\n\n" + r"\clearpage" + "\n" + r"\section{Apéndice: Preparación de Datos y Estimación de Parámetros}" + "\n\n" + body_01 + "\n\n" + end_marker
        
        with open("docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.tex", "w", encoding="utf-8") as f:
            f.write(new_content_02)
        print("done_appending_01_to_02")
    else:
        print("end document not found in 02")
else:
    print("markers not found in 01")
