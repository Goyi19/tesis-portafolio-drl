import zlib
import base64
import urllib.request
import os

def kroki_encode(text):
    compressed = zlib.compress(text.encode('utf-8'), 9)
    return base64.urlsafe_b64encode(compressed).decode('ascii')

files = [
    "docs/Guias Tecnicas DRL/diagrama_ppo_matematico.md",
    "docs/Guias Tecnicas DRL/diagrama_sac_matematico.md",
    "docs/Guias Tecnicas DRL/diagrama_drl_matematico.md"
]

for f in files:
    if not os.path.exists(f):
        print(f"Archivo no encontrado: {f}")
        continue
        
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Limpiar las etiquetas de markdown para dejar solo la sintaxis mermaid
    content = content.replace('```mermaid', '').replace('```', '').strip()
    
    encoded = kroki_encode(content)
    url = f"https://kroki.io/mermaid/svg/{encoded}" # Usaremos SVG para mejor calidad al verlo o PNG
    url_png = f"https://kroki.io/mermaid/png/{encoded}"
    
    out_name = f.replace('.md', '.png')
    
    print(f"Descargando {out_name} desde Kroki API...")
    try:
        # Se necesita un User-Agent a veces
        req = urllib.request.Request(url_png, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(out_name, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Guardado exitosamente: {out_name}")
    except Exception as e:
        print(f"Error procesando {f}: {e}")
