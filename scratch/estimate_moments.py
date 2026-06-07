import pandas as pd
import numpy as np

file_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\source_data.csv'
df = pd.read_csv(file_path)

# Inicializar columnas
assets = ['spx', 'cmc']
regimes = ['bear', 'bull']

for asset in assets:
    for regime in regimes:
        df[f'mu_{asset}_{regime}'] = 0.0
        df[f'var_{asset}_{regime}'] = 0.0

df['cov_bear'] = 0.0
df['cov_bull'] = 0.0

# Cálculo expansivo (Rolling Horizon / Cumulative)
for t_idx in range(len(df)):
    t_curr = t_idx + 1
    df_upto_t = df.iloc[:t_curr]
    
    for regime in regimes:
        # Suma de pesos (probabilidades) hasta t
        sum_p_spx = df_upto_t[f'p_{regime}_spx'].sum()
        sum_p_cmc = df_upto_t[f'p_{regime}_cmc'].sum()
        
        # 1. Medias Ponderadas
        mu_spx = (df_upto_t['ret_spx'] * df_upto_t[f'p_{regime}_spx']).sum() / sum_p_spx
        mu_cmc = (df_upto_t['ret_cmc'] * df_upto_t[f'p_{regime}_cmc']).sum() / sum_p_cmc
        
        df.at[t_idx, f'mu_spx_{regime}'] = mu_spx
        df.at[t_idx, f'mu_cmc_{regime}'] = mu_cmc
        
        # 2. Varianzas y Covarianzas Ponderadas
        # Usamos la formula: sum( P * (R_i - mu_i) * (R_j - mu_j) ) / sum(P)
        diff_spx = df_upto_t['ret_spx'] - mu_spx
        diff_cmc = df_upto_t['ret_cmc'] - mu_cmc
        
        var_spx = (df_upto_t[f'p_{regime}_spx'] * (diff_spx**2)).sum() / sum_p_spx
        var_cmc = (df_upto_t[f'p_{regime}_cmc'] * (diff_cmc**2)).sum() / sum_p_cmc
        
        # Para la covarianza, surge la duda: ¿qué probabilidad usar?
        # En HMM multivariado, se asume un regimen global o se promedian las marginales.
        # Dado que tenemos p_bear_spx y p_bear_cmc por separado, usaremos el promedio como proxy del regimen conjunto.
        p_joint = (df_upto_t[f'p_{regime}_spx'] + df_upto_t[f'p_{regime}_cmc']) / 2
        cov = (p_joint * diff_spx * diff_cmc).sum() / p_joint.sum()
        
        df.at[t_idx, f'var_spx_{regime}'] = var_spx
        df.at[t_idx, f'var_cmc_{regime}'] = var_cmc
        df.at[t_idx, f'cov_{regime}'] = cov

# Guardar resultados
df.to_csv(file_path, index=False)
print("Estimación de momentos por régimen completada.")
print(df.tail(5)[['t', 'mu_spx_bear', 'mu_spx_bull', 'var_spx_bear', 'cov_bear']])
