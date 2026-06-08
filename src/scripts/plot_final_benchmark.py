import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC, PPO
import sys

# Agregar src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_handler import DataHandler

def simulate_weights_trajectory(df_partition, weights_array, initial_capital=10000.0, c_base=[0.001, 0.004], c_mult=0.0):
    """
    Simula el capital final dado un arreglo de pesos (N_steps, 2), la data de mercado y los costos.
    """
    capital = initial_capital
    cap_history = [capital]
    current_weights = np.array([0.5, 0.5]) # asume inicia 50/50
    
    for i in range(len(df_partition)):
        # Pesos objetivo en paso i
        w_t = weights_array[i]
        
        # Rotacin y costos
        delta_w = w_t - current_weights
        costos_t = np.sum(np.abs(delta_w) * np.array(c_base) * c_mult)
        
        # Retornos
        R_t = np.array([df_partition.iloc[i]['ret_spx'], df_partition.iloc[i]['ret_cmc']])
        port_ret = np.sum(w_t * R_t)
        
        # Actualizar capital
        capital = capital * (1.0 + port_ret - costos_t)
        cap_history.append(capital)
        
        # Actualizar pesos
        current_weights = w_t
        
    return np.array(cap_history)

def main():
    print("Iniciando integracin de resultados GAMS vs DRL en TEST...")
    
    out_dir = Path('results/figures/final_benchmark')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Datos Test
    handler = DataHandler()
    df_test = handler.get_partition('test')
    test_dates = df_test['t'].values
    x_weeks = range(len(test_dates) + 1)
    
    # 2. Cargar Pesos de GAMS
    gams_weights_path = Path('data/Gams_results/gams_opt_weights.csv')
    if not gams_weights_path.exists():
        print(f"Error: No se encontr {gams_weights_path}")
        return
        
    gams_df = pd.read_csv(gams_weights_path)
    
    # Convertir test_dates (int) a formato string de GAMS ('t1', 't2', etc.)
    test_dates_str = [f't{x}' for x in test_dates]
    
    # Filtrar GAMS solo para las fechas de TEST
    gams_test = gams_df[gams_df['t'].isin(test_dates_str)].copy()
    
    # Extraer combinacin representativa de GAMS (p.ej. Lambda=0.7, Costo=0.2 (Media))
    l_val = 0.7
    c_val = 0.2
    gams_subset = gams_test[(gams_test['Lambda'] == l_val) & (gams_test['Costo_Mult'] == c_val)]
    
    # Pivotear pesos GAMS
    piv = gams_subset.pivot(index='t', columns='i', values='peso').loc[test_dates_str].fillna(0)
    gams_w_array = piv[['SPX', 'CMC200']].values
    
    # Simular GAMS
    gams_cap_curve = simulate_weights_trajectory(df_test, gams_w_array, c_mult=c_val)
    
    # 3. Simular Buy & Hold (50/50 estǭtico sin rebalanceo, no tiene costos de rotacin)
    bh_cap = 10000.0
    bh_caps = [bh_cap]
    w_bh = np.array([0.5, 0.5])
    for i in range(len(df_test)):
        R_t = np.array([df_test.iloc[i]['ret_spx'], df_test.iloc[i]['ret_cmc']])
        bh_cap *= (1.0 + np.sum(w_bh * R_t))
        bh_caps.append(bh_cap)
        w_bh = w_bh * (1.0 + R_t)
        w_bh /= np.sum(w_bh) if np.sum(w_bh) > 0 else 1.0
        
    # 4. Simular agente SAC (Lambda=0.7, Friccin=Media)
    # Recrear el entorno para cargar el modelo
    from environment.portfolio_env import PortfolioEnv
    env = PortfolioEnv(df_test, initial_capital=10000.0, transaction_cost_pct=[0.00020, 0.00080], lambda_risk=0.7, is_eval=True)
    sac_model_path = "models/sac_risk_cost_sens/Lambda_0p7_Friccion_Media/best_model.zip"
    
    sac_caps = [10000.0]
    if os.path.exists(sac_model_path):
        model = SAC.load(sac_model_path)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            sac_caps.append(info['capital'])
            done = terminated or truncated
    else:
        print(f"Advertencia: Modelo SAC no encontrado en {sac_model_path}")
        sac_caps = [np.nan] * len(x_weeks)
        
    # 5. GRAFICAR TODO JUNTO
    plt.figure(figsize=(14, 7))
    plt.plot(x_weeks, gams_cap_curve, label=f'GAMS Optimo (Terico L=0.7, Friccin Media) - Cap: ${gams_cap_curve[-1]:.0f}', color='green', linewidth=3)
    
    if not np.isnan(sac_caps[-1]):
        plt.plot(x_weeks, sac_caps, label=f'Agente SAC (L=0.7, Friccin Media) - Cap: ${sac_caps[-1]:.0f}', color='blue', linewidth=3)
        
    plt.plot(x_weeks, bh_caps, label=f'Benchmark Buy&Hold (50/50) - Cap: ${bh_caps[-1]:.0f}', color='red', linestyle='--', linewidth=2)
    
    plt.title("Rendimiento en TEST (Out-of-Sample): DRL vs GAMS vs Market", fontsize=16)
    plt.xlabel("Semanas (Particin Test)")
    plt.ylabel("Capital ($)")
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = out_dir / "Test_Capital_DRL_vs_GAMS.png"
    plt.savefig(plot_path)
    print(f"Grafico final exportado a: {plot_path}")
    
    # Tambien podemos exportar un CSV consolidado
    df_export = pd.DataFrame({
        'Semana_Test': list(range(len(x_weeks))),
        'B_H_Cap': bh_caps,
        'GAMS_Cap': gams_cap_curve,
        'SAC_Cap': sac_caps
    })
    csv_path = out_dir / "Data_Final_Test.csv"
    df_export.to_csv(csv_path, index=False)
    print(f"Datos consolidados exportados a: {csv_path}")

if __name__ == "__main__":
    main()
