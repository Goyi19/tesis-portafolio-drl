import sys

pdf_path = "docs/Registro de Definiciones del Proyecto/02_MDP_y_Modelo_Matematico.pdf"
out_path = "scratch/pdf_text.txt"

try:
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Extracted via fitz.")
except ImportError:
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted via PyPDF2.")
    except ImportError:
        print("No PyMuPDF or PyPDF2 found. Trying pdfminer.")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "PyMuPDF"])
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Installed and extracted via fitz.")
