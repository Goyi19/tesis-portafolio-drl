import pandas as pd

file_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\source_data.csv'
df = pd.read_csv(file_path)

# Mapeo de nombres para incluir "est" (estimado)
rename_dict = {
    'mu_spx_bear': 'mu_est_spx_bear',
    'mu_spx_bull': 'mu_est_spx_bull',
    'mu_cmc_bear': 'mu_est_cmc_bear',
    'mu_cmc_bull': 'mu_est_cmc_bull',
    'var_spx_bear': 'var_est_spx_bear',
    'var_spx_bull': 'var_est_spx_bull',
    'var_cmc_bear': 'var_est_cmc_bear',
    'var_cmc_bull': 'var_est_cmc_bull',
    'cov_bear': 'cov_est_bear',
    'cov_bull': 'cov_est_bull'
}

df.rename(columns=rename_dict, inplace=True)

# Guardar
df.to_csv(file_path, index=False)
print("Encabezados actualizados con el rasgo 'est' (estimado).")
print(df.columns)
