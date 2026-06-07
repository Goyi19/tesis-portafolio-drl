import pandas as pd
import re
import os

file_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\gams\ps.gms'

with open(file_path, 'r') as f:
    content = f.read()

def extract_table(content, table_name, num_cols):
    pattern = rf'TABLE {table_name}.*?\n(.*?);'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        # Try PARAMETER format if TABLE fails
        pattern = rf'PARAMETER {table_name}.*?\n/(.*?)/'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None
    
    lines = match.group(1).strip().split('\n')
    data = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= num_cols + 1:
            t = parts[0]
            values = [float(p) for p in parts[1:]]
            data[t] = values
    return data

def extract_parameter(content, param_name):
    pattern = rf'PARAMETER {param_name}.*?\n/(.*?)/'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    
    lines = match.group(1).strip().split('\n')
    data = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            t = parts[0]
            val = float(parts[1])
            data[t] = val
    return data

# Extraer datos
prob_spx = extract_table(content, 'prob_spx', 2) # bear, bull
prob_cmc = extract_table(content, 'prob_cmc200', 2) # bear, bull
ret_spx = extract_parameter(content, 'ret_semanal_spx')
ret_cmc = extract_parameter(content, 'ret_semanal_cmc200')

# Sincronizar por t
timesteps = [f't{i}' for i in range(1, 164)]
rows = []

for t in timesteps:
    row = {
        't': int(t[1:]),
        'ret_spx': ret_spx.get(t),
        'ret_cmc': ret_cmc.get(t),
        'p_bear_spx': prob_spx.get(t)[0] if prob_spx.get(t) else None,
        'p_bull_spx': prob_spx.get(t)[1] if prob_spx.get(t) else None,
        'p_bear_cmc': prob_cmc.get(t)[0] if prob_cmc.get(t) else None,
        'p_bull_cmc': prob_cmc.get(t)[1] if prob_cmc.get(t) else None
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Validaciones
print(f"Dataset shape: {df.shape}")
print(f"Null values:\n{df.isnull().sum()}")
print(f"Probabilities sum test (SPX t1): {df.iloc[0]['p_bear_spx'] + df.iloc[0]['p_bull_spx']}")

output_path = r'c:\Users\Rodrigo\.gemini\antigravity\scratch\tesis_portafolios_drl\data\source_data.csv'
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
