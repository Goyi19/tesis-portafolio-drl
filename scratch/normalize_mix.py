import pandas as pd
import numpy as np

file_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\source_data.csv'
df = pd.read_csv(file_path)

mix_cols = ['mu_mix_spx', 'mu_mix_cmc', 'var_mix_spx', 'var_mix_cmc', 'cov_mix']

# Normalización Z-Score Expansiva (Rolling/Expanding)
# z_t = (x_t - mean(x_{1..t})) / std(x_{1..t})
for col in mix_cols:
    # Calculamos media y std expansiva
    expanding_mean = df[col].expanding().mean()
    expanding_std = df[col].expanding().std()
    
    # Aplicamos formula (llenamos NaN iniciales con 0)
    df[f'z_{col}'] = (df[col] - expanding_mean) / expanding_std
    df[f'z_{col}'] = df[f'z_{col}'].fillna(0)

# Guardar resultados
df.to_csv(file_path, index=False)
print("Normalización Z-Score Expansiva completada.")
print(df.tail(5)[['t', 'z_mu_mix_spx', 'z_var_mix_spx', 'z_cov_mix']])
