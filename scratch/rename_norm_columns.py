import pandas as pd

file_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\source_data.csv'
df = pd.read_csv(file_path)

# Renombrar columnas z_ a norm_
new_columns = {col: col.replace('z_', 'norm_') for col in df.columns if col.startswith('z_')}
df.rename(columns=new_columns, inplace=True)

df.to_csv(file_path, index=False)
print("Prefijos actualizados de 'z_' a 'norm_'.")
print(df.columns)
