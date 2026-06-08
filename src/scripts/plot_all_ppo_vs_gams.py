import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
import sys

# Agregar src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_handler import DataHandler
from environment.portfolio_env import PortfolioEnv

def simulate_weights_trajectory(df_partition, weights_array, initial_capital=10000.0, c_base=[0.001, 0.004], c_mult=0.0):
    capital = initial_capital
    cap_history = [capital]
    current_weights = np.array([0.5, 0.5])
    for i in range(len(df_partition)):
        w_t = weights_array[i]
        delta_w = w_t - current_weights
        
        # Ojo: si c_mult es 'Base', el multiplicador no se usa, el base es 0.005 y 0.010 directo
        if c_mult == 'Base':
            costos_t = np.sum(np.abs(delta_w) * np.array([0.005, 0.010]))
        else:
            costos_t = np.sum(np.abs(delta_w) * np.array(c_base) * c_mult)
            
        R_t = np.array([df_partition.iloc[i]['ret_spx'], df_partition.iloc[i]['ret_cmc']])
        port_ret = np.sum(w_t * R_t)
        capital = capital * (1.0 + port_ret - costos_t)
        cap_history.append(capital)
        current_weights = w_t
    return np.array(cap_history)

def main():
    print("Iniciando simulacin de 10 PPO vs 10 GAMS en TEST...")
    out_dir = Path('results/figures/final_benchmark')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    handler = DataHandler()
    df_test = handler.get_partition('test')
    test_dates = df_test['t'].values
    test_dates_str = [f't{x}' for x in test_dates]
    x_weeks = range(len(test_dates) + 1)
    
    # ---------------------------------------------------------
    # 1. CARGAR 10 MODELOS GAMS
    # ---------------------------------------------------------
    gams_curves = {}
    
    # Sensibilidad GAMS
    gams_sens = pd.read_csv('data/Gams_results/gams_opt_weights.csv')
    gams_sens_test = gams_sens[gams_sens['t'].isin(test_dates_str)].copy()
    
    lambdas = [0.3, 0.7, 1.1]
    costos = [0.01, 0.2, 0.5]
    
    for l in lambdas:
        for c in costos:
            sub = gams_sens_test[(gams_sens_test['Lambda'] == l) & (gams_sens_test['Costo_Mult'] == c)]
            piv = sub.pivot(index='t', columns='i', values='peso').loc[test_dates_str].fillna(0)
            w_arr = piv[['SPX', 'CMC200']].values
            cap_curve = simulate_weights_trajectory(df_test, w_arr, c_mult=c)
            gams_curves[f"GAMS_L{l}_C{c}"] = cap_curve
            
    # Base GAMS
    gams_base = pd.read_csv('data/Gams_results/gams_opt_weights_base.csv')
    gams_base_test = gams_base[gams_base['t'].isin(test_dates_str)].copy()
    piv = gams_base_test.pivot(index='t', columns='i', values='peso').loc[test_dates_str].fillna(0)
    w_arr = piv[['SPX', 'CMC200']].values
    cap_curve = simulate_weights_trajectory(df_test, w_arr, c_mult='Base')
    gams_curves["GAMS_Base"] = cap_curve
    
    # ---------------------------------------------------------
    # 2. CARGAR 10 MODELOS PPO
    # ---------------------------------------------------------
    ppo_curves = {}
    
    friction_map = {'Baja': [0.00001, 0.00004], 'Media': [0.00020, 0.00080], 'Alta': [0.00050, 0.00200]}
    lambda_map = {0.3: '0p3', 0.7: '0p7', 1.1: '1p1'}
    
    def simulate_agent(model_path, env):
        caps = [10000.0]
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                caps.append(info['capital'])
                done = terminated or truncated
        else:
            print(f"No se encontr {model_path}")
            caps = [np.nan] * len(x_weeks)
        return np.array(caps)
        
    # Sensibilidad PPO
    for l in lambdas:
        for f_name, f_val in friction_map.items():
            l_str = lambda_map[l]
            m_path = f"models/ppo_risk_cost_sens/Lambda_{l_str}_Friccion_{f_name}/best_model.zip"
            env = PortfolioEnv(df_test, transaction_cost_pct=f_val, lambda_risk=l, is_eval=True)
            ppo_curves[f"PPO_L{l}_{f_name}"] = simulate_agent(m_path, env)
            
    # Base PPO
    m_path = "models/ppo/best_model/best_model.zip"
    env = PortfolioEnv(df_test, transaction_cost_pct=[0.005, 0.010], lambda_risk=0.10, is_eval=True)
    ppo_curves["PPO_Base"] = simulate_agent(m_path, env)
    
    # ---------------------------------------------------------
    # 3. GRAFICAR
    # ---------------------------------------------------------
    plt.figure(figsize=(16, 9))
    
    # Plot GAMS (Techo terico - Sombras de Verde)
    gams_colors = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
    for idx, (name, curve) in enumerate(gams_curves.items()):
        lw = 4 if name == "GAMS_Base" else 1.5
        alp = 1.0 if name == "GAMS_Base" else 0.4
        lbl = "GAMS (Base)" if name == "GAMS_Base" else ("GAMS (Sensibilidad)" if idx==0 else "")
        plt.plot(x_weeks, curve, color=gams_colors[idx], linewidth=lw, alpha=alp, label=lbl)

    # Plot PPO (Nuestros modelos - Sombras de Azul/Morado)
    ppo_colors = plt.cm.Purples(np.linspace(0.5, 1.0, 10))
    for idx, (name, curve) in enumerate(ppo_curves.items()):
        lw = 4 if name == "PPO_Base" else 2.0
        alp = 1.0 if name == "PPO_Base" else 0.6
        lbl = "PPO (Base)" if name == "PPO_Base" else ("PPO (Sensibilidad)" if idx==0 else "")
        plt.plot(x_weeks, curve, color=ppo_colors[idx], linewidth=lw, alpha=alp, label=lbl)
        
    # Benchmarks de mercado (Gris y Rojo)
    bh_cap = 10000.0
    bh_caps = [bh_cap]
    w_bh = np.array([0.5, 0.5])
    for i in range(len(df_test)):
        R_t = np.array([df_test.iloc[i]['ret_spx'], df_test.iloc[i]['ret_cmc']])
        bh_cap *= (1.0 + np.sum(w_bh * R_t))
        bh_caps.append(bh_cap)
        w_bh = w_bh * (1.0 + R_t)
        w_bh /= np.sum(w_bh) if np.sum(w_bh) > 0 else 1.0
        
    plt.plot(x_weeks, bh_caps, color='red', linestyle='--', linewidth=3, label="Benchmark: Buy & Hold")

    plt.title("Evolucin de Capital en TEST: Los 10 Modelos PPO vs 10 Modelos GAMS", fontsize=18, fontweight='bold')
    plt.xlabel("Semanas (Conjunto Test)", fontsize=14)
    plt.ylabel("Capital Final ($)", fontsize=14)
    plt.grid(alpha=0.3)
    
    # Ajustar leyenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=12)
    
    plot_path = out_dir / "Test_Capital_ALL_PPO_vs_GAMS.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Grfico exportado con xito a: {plot_path}")

if __name__ == "__main__":
    main()
