import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.ppo_agent import PPOPortfolioAgent

def calc_metrics(caps, rewards, actions):
    caps_ary = np.array(caps)
    returns = np.diff(caps_ary) / caps_ary[:-1]
    
    # Recompensa Total del Episodio
    total_reward = np.sum(rewards) if len(rewards) > 0 else 0.0
    
    # Sharpe Ratio (Anualizado, asumiendo 52 semanas por año)
    if np.std(returns) == 0:
        sharpe = 0.0
    else:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(52)
        
    # Max Drawdown
    roll_max = np.maximum.accumulate(caps_ary)
    drawdowns = (roll_max - caps_ary) / roll_max
    max_dd = np.max(drawdowns) * 100
    
    # Tasa de rotación (Turnover) promedio semanal
    actions_ary = np.array(actions)
    if len(actions_ary) > 1:
        turnover = np.mean(np.sum(np.abs(np.diff(actions_ary, axis=0)), axis=1)) * 100
    else:
        turnover = 0.0
        
    return total_reward, sharpe, max_dd, turnover

def extract_metric(event_acc, metric_name):
    if metric_name not in event_acc.Tags()['scalars']: return [], []
    events = event_acc.Scalars(metric_name)
    return [e.step for e in events], [e.value for e in events]

def main():
    models_dir = "models/ppo_sens"
    log_dir = "logs/ppo_sens_tensorboard"
    
    if not os.path.exists(models_dir):
        print(f"Directorio {models_dir} no existe. Ejecuta run_sensitivity.py primero.")
        return
        
    # Obtener los nombres EXACTOS de los experimentos leyendo las carpetas donde se guardaron los modelos
    # (SB3 no añade "_1" a estas carpetas, así que los nombres son puros e inmutables)
    exp_names = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))])
    
    if not exp_names:
        print("No hay modelos entrenados en la carpeta.")
        return
    
    # 1. Gráficos Comparativos de TensorBoard
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.canvas.manager.set_window_title('Sensibilidad: Redes Neuronales (TensorBoard)')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
    
    for idx, exp_name in enumerate(exp_names):
        # Tensorboard podría haberle añadido "_1", así que usamos glob para buscarlo usando el nombre puro como prefijo
        tb_dirs = glob.glob(os.path.join(log_dir, f"{exp_name}_*"))
        if not tb_dirs:
            # Por si acaso no le puso "_1"
            tb_dirs = [os.path.join(log_dir, exp_name)] if os.path.exists(os.path.join(log_dir, exp_name)) else []
            
        if not tb_dirs: 
            continue
            
        subdir = tb_dirs[0] # Tomamos la primera coincidencia
        event_files = glob.glob(os.path.join(subdir, "events.out.tfevents.*"))
        if not event_files: continue
        
        acc = EventAccumulator(event_files[0])
        acc.Reload()
        
        steps, rew = extract_metric(acc, 'rollout/ep_rew_mean')
        if rew: axs1[0, 0].plot(steps, rew, label=exp_name, color=colors[idx])
        
        steps, vloss = extract_metric(acc, 'train/value_loss')
        if vloss: axs1[0, 1].plot(steps, vloss, label=exp_name, color=colors[idx])
        
        steps, ploss = extract_metric(acc, 'train/policy_gradient_loss')
        if ploss: axs1[1, 0].plot(steps, ploss, label=exp_name, color=colors[idx])
        
        steps, ent = extract_metric(acc, 'train/entropy_loss')
        if ent: axs1[1, 1].plot(steps, ent, label=exp_name, color=colors[idx])
        
    axs1[0, 0].set_title('Recompensa Media'); axs1[0, 0].legend()
    axs1[0, 1].set_title('Value Loss'); axs1[0, 1].legend()
    axs1[1, 0].set_title('Policy Loss'); axs1[1, 0].legend()
    axs1[1, 1].set_title('Entropy Loss'); axs1[1, 1].legend()
    plt.tight_layout()
    
    # 2. Gráfico Comparativo de Evolución del Capital (TRAIN)
    print("Simulando capitales para todos los experimentos...")
    handler = DataHandler()
    train_df = handler.get_partition('train')
    val_df = handler.get_partition('val')
    
    fig2, ax2_train = plt.subplots(figsize=(12, 6))
    fig2.canvas.manager.set_window_title('Sensibilidad: Capital (TRAIN)')
    
    fig3, ax2_val = plt.subplots(figsize=(12, 6))
    fig3.canvas.manager.set_window_title('Sensibilidad: Capital (VALIDACIÓN)')
    
    colors_cap = plt.cm.tab20(np.linspace(0, 1, len(exp_names)))
    
    # Diccionarios para guardar las métricas calculadas
    metrics_train = {}
    metrics_val = {}
    
    # Eje X base para validación (comienza donde termina train)
    x_offset_val = len(train_df)
    
    for idx, exp_name in enumerate(exp_names):
        model_path = os.path.join(models_dir, exp_name, "best_model", "best_model.zip")
        if not os.path.exists(model_path): continue
            
        lambda_val = 1.0 if exp_name == "Risk_High" else (0.0 if exp_name == "Risk_Low" else 0.10)
        tc_val = [0.0, 0.0] if exp_name == "Cost_Zero" else ([0.02, 0.04] if exp_name == "Cost_High" else [0.005, 0.010])
        
        # --- SIMULAR EN TRAIN ---
        env_train = PortfolioEnv(df=train_df, initial_capital=10000.0, transaction_cost_pct=tc_val, lambda_risk=lambda_val, is_eval=True)
        agent_train = PPOPortfolioAgent(env=env_train, verbose=0)
        agent_train.load(model_path[:-4])
        
        obs, _ = env_train.reset()
        caps_train = [10000.0]
        rewards_train = []
        actions_train = []
        term = trunc = False
        while not (term or trunc):
            action = agent_train.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env_train.step(action)
            caps_train.append(info['capital'])
            rewards_train.append(reward)
            actions_train.append(action)
            
        metrics_train[exp_name] = calc_metrics(caps_train, rewards_train, actions_train) + (caps_train[-1],)
        ax2_train.plot(range(len(caps_train)), caps_train, label=exp_name, color=colors_cap[idx], linewidth=2)

        # --- SIMULAR EN VAL ---
        env_val = PortfolioEnv(df=val_df, initial_capital=10000.0, transaction_cost_pct=tc_val, lambda_risk=lambda_val, is_eval=True)
        agent_val = PPOPortfolioAgent(env=env_val, verbose=0)
        agent_val.load(model_path[:-4])
        
        obs, _ = env_val.reset()
        caps_val = [10000.0]
        rewards_val = []
        actions_val = []
        term = trunc = False
        while not (term or trunc):
            action = agent_val.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env_val.step(action)
            caps_val.append(info['capital'])
            rewards_val.append(reward)
            actions_val.append(action)
            
        metrics_val[exp_name] = calc_metrics(caps_val, rewards_val, actions_val) + (caps_val[-1],)
        x_val = np.arange(x_offset_val, x_offset_val + len(caps_val))
        ax2_val.plot(x_val, caps_val, label=exp_name, color=colors_cap[idx], linewidth=2)
        
    # --- AÑADIR NAIVE BUY & HOLD (TRAIN) ---
    cap_bh_train = 10000.0; caps_bh_train = [cap_bh_train]; w_bh_t = np.array([0.5, 0.5])
    for i in range(len(train_df)):
        R_t = np.array([train_df.iloc[i]['ret_spx'], train_df.iloc[i]['ret_cmc']])
        cap_bh_train = cap_bh_train * (1.0 + np.sum(w_bh_t * R_t))
        caps_bh_train.append(cap_bh_train)
        w_bh_t = w_bh_t * (1.0 + R_t); w_bh_t = w_bh_t / np.sum(w_bh_t) if np.sum(w_bh_t) > 0 else w_bh_t
    
    # Calcular métricas para B&H Train
    returns_bh_train = np.diff(caps_bh_train) / np.array(caps_bh_train)[:-1]
    sharpe_bh_train = (np.mean(returns_bh_train) / np.std(returns_bh_train)) * np.sqrt(52) if np.std(returns_bh_train) > 0 else 0
    roll_max_bh = np.maximum.accumulate(caps_bh_train)
    mdd_bh_train = np.max((roll_max_bh - caps_bh_train) / roll_max_bh) * 100
    metrics_train["Naive B&H"] = (0.0, sharpe_bh_train, mdd_bh_train, 0.0, caps_bh_train[-1])
    
    ax2_train.plot(range(len(caps_bh_train)), caps_bh_train, label="Naive B&H 50/50", color='black', linestyle='--', linewidth=3)
    ax2_train.axhline(10000, color='red', linestyle=':')
    ax2_train.set_title("Evolución del Capital - Set de ENTRENAMIENTO")
    ax2_train.set_xlabel("Semanas")
    ax2_train.set_ylabel("Capital ($)")
    ax2_train.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2_train.grid(True, alpha=0.3)
    fig2.tight_layout()
    
    # --- AÑADIR NAIVE BUY & HOLD (VAL) ---
    cap_bh_val = 10000.0; caps_bh_val = [cap_bh_val]; w_bh_v = np.array([0.5, 0.5])
    for i in range(len(val_df)):
        R_t = np.array([val_df.iloc[i]['ret_spx'], val_df.iloc[i]['ret_cmc']])
        cap_bh_val = cap_bh_val * (1.0 + np.sum(w_bh_v * R_t))
        caps_bh_val.append(cap_bh_val)
        w_bh_v = w_bh_v * (1.0 + R_t); w_bh_v = w_bh_v / np.sum(w_bh_v) if np.sum(w_bh_v) > 0 else w_bh_v
        
    returns_bh_val = np.diff(caps_bh_val) / np.array(caps_bh_val)[:-1]
    sharpe_bh_val = (np.mean(returns_bh_val) / np.std(returns_bh_val)) * np.sqrt(52) if np.std(returns_bh_val) > 0 else 0
    roll_max_bh_val = np.maximum.accumulate(caps_bh_val)
    mdd_bh_val = np.max((roll_max_bh_val - caps_bh_val) / roll_max_bh_val) * 100
    metrics_val["Naive B&H"] = (0.0, sharpe_bh_val, mdd_bh_val, 0.0, caps_bh_val[-1])
    
    x_val_bh = np.arange(x_offset_val, x_offset_val + len(caps_bh_val))
    ax2_val.plot(x_val_bh, caps_bh_val, label="Naive B&H 50/50", color='black', linestyle='--', linewidth=3)
    ax2_val.axhline(10000, color='red', linestyle=':')
    ax2_val.set_title("Evolución del Capital - Set de VALIDACIÓN")
    ax2_val.set_xlabel("Semanas")
    ax2_val.set_ylabel("Capital ($)")
    ax2_val.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2_val.grid(True, alpha=0.3)
    fig3.tight_layout()
    
    # IMPRIMIR TABLAS DE MÉTRICAS EN CONSOLA
    def print_table(title, metrics_dict):
        print(f"\n{'='*90}")
        print(f"{title:^90}")
        print(f"{'='*90}")
        print(f"{'Modelo':<15} | {'Cap. Final ($)':<14} | {'Recompensa':<10} | {'Sharpe':<8} | {'Max DD (%)':<10} | {'Rotación (%)':<12}")
        print("-" * 90)
        # Ordenar por Capital Final descendente
        for name, m in sorted(metrics_dict.items(), key=lambda x: x[1][4], reverse=True):
            print(f"{name:<15} | ${m[4]:<13.2f} | {m[0]:<10.4f} | {m[1]:<8.2f} | {m[2]:<10.2f} | {m[3]:<12.2f}")
        print("-" * 90)

    print_table("MÉTRICAS SET DE ENTRENAMIENTO (TRAIN)", metrics_train)
    print_table("MÉTRICAS SET DE VALIDACIÓN (VAL)", metrics_val)
    
    print("\nMostrando resultados en ventanas separadas...")
    plt.show()

if __name__ == "__main__":
    main()
