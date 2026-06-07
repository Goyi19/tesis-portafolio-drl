import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.ppo_agent import PPOPortfolioAgent

def calc_metrics(caps, rewards, actions):
    caps_ary = np.array(caps)
    returns = np.diff(caps_ary) / caps_ary[:-1]
    total_reward = np.sum(rewards) if len(rewards) > 0 else 0.0
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(52) if np.std(returns) > 0 else 0.0
    roll_max = np.maximum.accumulate(caps_ary)
    max_dd = np.max((roll_max - caps_ary) / roll_max) * 100
    actions_ary = np.array(actions)
    if len(actions_ary) > 1:
        turnover = np.mean(np.sum(np.abs(np.diff(actions_ary, axis=0)), axis=1)) * 100
    else:
        turnover = 0.0
    return total_reward, sharpe, max_dd, turnover, caps[-1]

def simulate_crp(df, tc_val=[0.005, 0.010]):
    """Simula un Modelo de Rebalanceo Constante (Equitativo 50/50)"""
    cap = 10000.0
    caps = [cap]
    w_target = np.array([0.5, 0.5])
    w_actual = np.array([0.5, 0.5])
    
    for i in range(len(df)):
        delta_w = w_target - w_actual
        cost = np.sum(np.abs(delta_w) * tc_val)
        cap = cap * (1.0 - cost)
        w_actual = w_target
        
        R_t = np.array([df.iloc[i]['ret_spx'], df.iloc[i]['ret_cmc']])
        cap = cap * (1.0 + np.sum(w_actual * R_t))
        
        w_actual = w_actual * (1.0 + R_t)
        w_actual = w_actual / np.sum(w_actual) if np.sum(w_actual) > 0 else np.array([0.5, 0.5])
        caps.append(cap)
        
    return caps

def extract_tb_metrics(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    
    metrics = {
        'reward': 'rollout/ep_rew_mean',
        'value_loss': 'train/value_loss',
        'policy_loss': 'train/policy_loss',
        'entropy': 'train/entropy_loss'
    }
    
    data = {}
    for key, tag in metrics.items():
        if tag in tags:
            events = event_acc.Scalars(tag)
            data[key] = ([e.step for e in events], [e.value for e in events])
    return data

def main():
    print("========================================")
    print(" EVALUACIÓN DEL MODELO BASE OFICIAL PPO ")
    print("========================================")
    
    handler = DataHandler()
    parts = {'train': handler.get_partition('train'), 'val': handler.get_partition('val'), 'test': handler.get_partition('test')}
    
    model_path = "models/ppo/best_model/best_model.zip"
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo base en {model_path}")
        print("Por favor, entrena primero ejecutando: python src/scripts/train_ppo.py")
        return

    out_dir = "results/figures/Modelo_Base"
    os.makedirs(out_dir, exist_ok=True)
    
    # -----------------------------------------------------
    # 1. MÉTRICAS DE TENSORBOARD (Entrenamiento de la Red)
    # -----------------------------------------------------
    log_dir = "logs/ppo_tensorboard"
    if os.path.exists(log_dir):
        # Tomar la carpeta más reciente que no sea de sensibilidad (suele llamarse PPO_1, PPO_2, etc sin "Lambda")
        subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        # Filtramos las de sensibilidad para agarrar la original
        base_logs = [d for d in subdirs if "Lambda" not in d]
        
        if base_logs:
            latest_log = max(base_logs, key=os.path.getmtime)
            print(f"Extrayendo métricas de Tensorboard desde: {latest_log}")
            tb_data = extract_tb_metrics(latest_log)
            
            fig_tb, axes_tb = plt.subplots(2, 2, figsize=(14, 10))
            fig_tb.suptitle("Evolución del Entrenamiento PPO (Modelo Base)", fontsize=18)
            
            titles = {'reward': 'Recompensa Media (ep_rew_mean)', 'value_loss': 'Loss Crítico (value_loss)', 
                      'policy_loss': 'Loss Actor (policy_loss)', 'entropy': 'Pérdida de Entropía'}
            colors = {'reward': 'blue', 'value_loss': 'red', 'policy_loss': 'orange', 'entropy': 'purple'}
            
            for k, (steps, vals) in tb_data.items():
                if not steps: continue
                ax = axes_tb[0 if k in ['reward', 'value_loss'] else 1, 0 if k in ['reward', 'policy_loss'] else 1]
                ax.plot(steps, vals, color=colors[k], linewidth=2)
                ax.set_title(titles[k])
                ax.set_xlabel("Timesteps")
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "Tensorboard_Metrics.png"))
            plt.close()
    
    # -----------------------------------------------------
    # 2. EVALUACIÓN POR PARTICIÓN
    # -----------------------------------------------------
    for p_name, df in parts.items():
        print(f"\n--- Evaluando en Set: {p_name.upper()} ---")
        
        # Eje temporal continuo para gráficos
        if p_name == 'train': x_weeks = np.arange(len(df)+1)
        elif p_name == 'val': x_weeks = np.arange(len(parts['train']), len(parts['train'])+len(df)+1)
        else: x_weeks = np.arange(len(parts['train'])+len(parts['val']), len(parts['train'])+len(parts['val'])+len(df)+1)
        
        # Instanciar Entorno Base (Lambda=0.10, TC=OFICIAL) e is_eval=True SIEMPRE para graficar cronológicamente
        env = PortfolioEnv(df=df, is_eval=True)
        agent = PPOPortfolioAgent(env=env, verbose=0)
        agent.load(model_path[:-4])
        
        obs, _ = env.reset(); caps = [10000.0]; w_spx = [0.5]; w_cmc = [0.5]; rews = []; acts = []
        v_preds = []
        
        term = trunc = False
        while not (term or trunc):
            action = agent.predict(obs, deterministic=True)
            
            obs_tensor, _ = agent.model.policy.obs_to_tensor(obs)
            v_s = agent.model.policy.predict_values(obs_tensor).item()
            v_preds.append(v_s)
            
            obs, r, term, trunc, info = env.step(action)
            caps.append(info['capital']); w_spx.append(info['weights_spx']); w_cmc.append(info['weights_cmc'])
            rews.append(r); acts.append(action)
            
        gamma = 0.99
        G_t = np.zeros(len(rews)); g = 0
        for t in reversed(range(len(rews))):
            g = rews[t] + gamma * g
            G_t[t] = g
            
        # Métricas PPO
        met_ppo = calc_metrics(caps, rews, acts)
        
        # Benchmarks
        caps_crp = simulate_crp(df)
        met_crp = calc_metrics(caps_crp, [], [ [0.5, 0.5] for _ in range(len(df)) ])
        
        cap_bh = 10000.0; caps_bh = [cap_bh]; w_bh = np.array([0.5, 0.5])
        for i in range(len(df)):
            R_t = np.array([df.iloc[i]['ret_spx'], df.iloc[i]['ret_cmc']])
            cap_bh *= (1.0 + np.sum(w_bh * R_t)); caps_bh.append(cap_bh)
            w_bh = w_bh * (1.0 + R_t); w_bh /= np.sum(w_bh) if np.sum(w_bh)>0 else 1.0
        met_bh = calc_metrics(caps_bh, [], [ [0.5, 0.5] for _ in range(len(df)) ])
        
        # a) EVOLUCIÓN DE CAPITAL
        plt.figure(figsize=(12, 7))
        plt.plot(x_weeks, caps, label="Agente PPO (Base)", color='blue', linewidth=2.5)
        plt.plot(x_weeks, caps_crp, label="Rebalanceo Constante (50/50)", color='green', linestyle='--', linewidth=2)
        plt.plot(x_weeks, caps_bh, label="Naive B&H (50/50)", color='black', linestyle=':', linewidth=2)
        plt.title(f"Evolución del Capital - {p_name.upper()}", fontsize=16)
        plt.xlabel("Semanas"); plt.ylabel("Capital ($)")
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Capital_{p_name}.png"))
        plt.close()
        
        # b) ESTRATEGIA (ÁREAS APILADAS)
        plt.figure(figsize=(12, 6))
        plt.stackplot(x_weeks, w_spx, w_cmc, labels=['SPX (S&P 500)', 'CMC200 (Cripto)'], colors=['#1f77b4', '#ff7f0e'], alpha=0.8)
        plt.title(f"Estrategia de Inversión a lo largo del tiempo - {p_name.upper()}", fontsize=16)
        plt.xlabel("Semanas"); plt.ylabel("Distribución de Pesos")
        plt.legend(loc='upper right', fontsize=12); plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Estrategia_Temporal_{p_name}.png"))
        plt.close()
        
        # c) HISTOGRAMA (BANG-BANG TEST)
        plt.figure(figsize=(10, 6))
        plt.hist(w_spx, bins=30, color='purple', alpha=0.7, edgecolor='black')
        plt.title(f"Histograma de Pesos SPX (Bang-Bang Test) - {p_name.upper()}", fontsize=16)
        plt.xlabel("Peso Asignado a SPX"); plt.ylabel("Frecuencia (Semanas)")
        plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Histograma_Pesos_{p_name}.png"))
        plt.close()
        
        # d) FUNCIÓN DE VALOR (CRÍTICO VS RETORNO REAL)
        plt.figure(figsize=(12, 6))
        plt.plot(x_weeks[:-1], v_preds, label="Predicción V(s) (El Crítico)", color='blue', linewidth=2)
        plt.plot(x_weeks[:-1], G_t, label="Retorno Real Descontado $G_t$", color='red', alpha=0.6, linestyle='--')
        plt.title(f"Ego del Crítico: Predicción vs Realidad - {p_name.upper()}", fontsize=16)
        plt.xlabel("Semanas"); plt.ylabel("Valor")
        plt.legend(loc='best'); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Value_Function_{p_name}.png"))
        plt.close()
        
        # GRÁFICOS EXCLUSIVOS PARA VAL Y TEST
        if p_name in ['val', 'test']:
            # e) SCATTER PLOT (CORDURA)
            if 'norm_var_mix_cmc' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df['norm_var_mix_cmc'].values, w_cmc[1:], alpha=0.7, color='teal', edgecolors='black', s=50)
                plt.title(f"Prueba de Cordura: Volatilidad vs Decisión - {p_name.upper()}", fontsize=16)
                plt.xlabel("Volatilidad Cripto Predicha (norm_var_mix_cmc)")
                plt.ylabel("Peso Asignado a CMC200")
                z = np.polyfit(df['norm_var_mix_cmc'].values, w_cmc[1:], 1); p = np.poly1d(z)
                plt.plot(df['norm_var_mix_cmc'].values, p(df['norm_var_mix_cmc'].values), "r--", linewidth=2, label="Tendencia")
                plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"Scatter_Feature_{p_name}.png"))
                plt.close()
                
            # f) ROLLING METRICS (ESTABILIDAD)
            window = min(12, max(4, len(df) // 4))
            returns = np.diff(caps) / caps[:-1]
            roll_sharpe = pd.Series(returns).rolling(window).apply(lambda x: (np.mean(x)/np.std(x))*np.sqrt(52) if np.std(x)>0 else 0)
            def calc_maxdd(x): return np.max((np.maximum.accumulate(x) - x) / np.maximum.accumulate(x)) * 100
            roll_maxdd = pd.Series(caps).rolling(window).apply(calc_maxdd)
            
            fig_r, ax_r1 = plt.subplots(figsize=(12, 6))
            ax_r1.plot(x_weeks[1:], roll_sharpe, color='green', label=f'Sharpe Ratio (Win={window})', linewidth=2)
            ax_r1.set_ylabel("Sharpe Ratio", color='green', fontsize=12)
            ax_r1.axhline(0, color='black', linewidth=0.8, linestyle=':')
            ax_r2 = ax_r1.twinx()
            ax_r2.plot(x_weeks, roll_maxdd, color='red', label=f'Max Drawdown % (Win={window})', linestyle='--', linewidth=2)
            ax_r2.set_ylabel("Max Drawdown (%)", color='red', fontsize=12)
            plt.title(f"Rolling Metrics de Estabilidad - {p_name.upper()}", fontsize=16)
            lines_1, labels_1 = ax_r1.get_legend_handles_labels(); lines_2, labels_2 = ax_r2.get_legend_handles_labels()
            ax_r1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Rolling_Metrics_{p_name}.png"))
            plt.close(fig_r)

        # TABLA EN CONSOLA
        print(f"{'Modelo':<25} | {'Cap. Final ($)':<14} | {'Recompensa':<10} | {'Sharpe':<8} | {'MaxDD(%)':<8} | {'Rotación(%)':<10}")
        print("-" * 90)
        print(f"{'PPO Base':<25} | ${met_ppo[4]:<13.2f} | {met_ppo[0]:<10.4f} | {met_ppo[1]:<8.2f} | {met_ppo[2]:<8.2f} | {met_ppo[3]:<10.2f}")
        print(f"{'Rebalanceo Constante (CRP)':<25} | ${met_crp[4]:<13.2f} | {met_crp[0]:<10.4f} | {met_crp[1]:<8.2f} | {met_crp[2]:<8.2f} | {met_crp[3]:<10.2f}")
        print(f"{'Naive B&H (50/50)':<25} | ${met_bh[4]:<13.2f} | {met_bh[0]:<10.4f} | {met_bh[1]:<8.2f} | {met_bh[2]:<8.2f} | {met_bh[3]:<10.2f}")
        print("-" * 90)

if __name__ == "__main__":
    main()
