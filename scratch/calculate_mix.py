import pandas as pd

file_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\source_data.csv'
df = pd.read_csv(file_path)

# 1. Medias Mix
df['mu_mix_spx'] = df['p_bear_spx'] * df['mu_est_spx_bear'] + df['p_bull_spx'] * df['mu_est_spx_bull']
df['mu_mix_cmc'] = df['p_bear_cmc'] * df['mu_est_cmc_bear'] + df['p_bull_cmc'] * df['mu_est_cmc_bull']

# 2. Varianzas Mix (diag de la covarianza mezclada)
df['var_mix_spx'] = df['p_bear_spx'] * df['var_est_spx_bear'] + df['p_bull_spx'] * df['var_est_spx_bull']
df['var_mix_cmc'] = df['p_bear_cmc'] * df['var_est_cmc_bear'] + df['p_bull_cmc'] * df['var_est_cmc_bull']

# 3. Covarianza Mix
# Usamos el promedio de probabilidades como peso conjunto para la mezcla de la covarianza
p_joint_bear = (df['p_bear_spx'] + df['p_bear_cmc']) / 2
p_joint_bull = (df['p_bull_spx'] + df['p_bull_cmc']) / 2

df['cov_mix'] = p_joint_bear * df['cov_est_bear'] + p_joint_bull * df['cov_est_bull']

# Guardar
df.to_csv(file_path, index=False)
print("Construcción de variables Mix completada.")
print(df.tail(5)[['t', 'mu_mix_spx', 'var_mix_spx', 'cov_mix']])
