import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.sac_agent import SACPortfolioAgent

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
        'critic_loss': 'train/critic_loss',
        'actor_loss': 'train/actor_loss',
        'ent_coef': 'train/ent_coef'
    }
    data = {}
    for key, tag in metrics.items():
        if tag in tags:
            events = event_acc.Scalars(tag)
            data[key] = ([e.step for e in events], [e.value for e in events])
    return data

def main():
    print("========================================")
    print(" EVALUACIÓN DEL MODELO BASE OFICIAL SAC ")
    print("========================================")
    
    handler = DataHandler()
    parts = {'train': handler.get_partition('train'), 'val': handler.get_partition('val'), 'test': handler.get_partition('test')}
    
    model_path = "models/sac/best_model/best_model.zip"
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo base en {model_path}")
        return

    out_dir = "results/figures/Modelo_Base_SAC"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. MÉTRICAS TENSORBOARD
    log_dir = "logs/sac_tensorboard"
    if os.path.exists(log_dir):
        subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if subdirs:
            latest_log = max(subdirs, key=os.path.getmtime)
            tb_data = extract_tb_metrics(latest_log)
            fig_tb, axes_tb = plt.subplots(2, 2, figsize=(14, 10))
            fig_tb.suptitle("Evolución del Entrenamiento SAC (Modelo Base)", fontsize=18)
            titles = {'reward': 'Recompensa Media (ep_rew_mean)', 'critic_loss': 'Loss Crítico (critic_loss)', 
                      'actor_loss': 'Loss Actor (actor_loss)', 'ent_coef': 'Coeficiente Entropía'}
            colors = {'reward': 'blue', 'critic_loss': 'red', 'actor_loss': 'orange', 'ent_coef': 'green'}
            for k, (steps, vals) in tb_data.items():
                if not steps: continue
                ax = axes_tb[0 if k in ['reward', 'critic_loss'] else 1, 0 if k in ['reward', 'actor_loss'] else 1]
                ax.plot(steps, vals, color=colors[k], linewidth=2)
                ax.set_title(titles[k])
                ax.set_xlabel("Timesteps")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "Tensorboard_Metrics.png"))
            plt.close()
    
    # 2. EVALUACIÓN POR PARTICIÓN
    for p_name, df in parts.items():
        print(f"\n--- Evaluando en Set: {p_name.upper()} ---")
        if p_name == 'train': x_weeks = np.arange(len(df)+1)
        elif p_name == 'val': x_weeks = np.arange(len(parts['train']), len(parts['train'])+len(df)+1)
        else: x_weeks = np.arange(len(parts['train'])+len(parts['val']), len(parts['train'])+len(parts['val'])+len(df)+1)
        
        env = PortfolioEnv(df=df, is_eval=True)
        agent = SACPortfolioAgent(env=env, verbose=0)
        agent.load(model_path[:-4])
        
        obs, _ = env.reset(); caps = [10000.0]; w_spx = [0.5]; w_cmc = [0.5]; rews = []; acts = []; q_preds = []
        term = trunc = False
        while not (term or trunc):
            action = agent.predict(obs, deterministic=True)
            
            # Predict Q(s, a) using SAC critic
            obs_tensor, _ = agent.model.policy.obs_to_tensor(obs)
            action_tensor = torch.tensor(action, dtype=torch.float32).reshape(1, -1).to(agent.model.device)
            with torch.no_grad():
                q_values = agent.model.critic(obs_tensor, action_tensor)
                # SAC uses 2 Q-networks, we take the minimum to be conservative
                q_s_a = torch.min(q_values[0], q_values[1]).item()
            q_preds.append(q_s_a)
            
            obs, r, term, trunc, info = env.step(action)
            caps.append(info['capital']); w_spx.append(info['weights_spx']); w_cmc.append(info['weights_cmc'])
            rews.append(r); acts.append(action)
            
        gamma = 0.99
        G_t = np.zeros(len(rews)); g = 0
        for t in reversed(range(len(rews))):
            g = rews[t] + gamma * g
            G_t[t] = g
            
        met_sac = calc_metrics(caps, rews, acts)
        caps_crp = simulate_crp(df)
        met_crp = calc_metrics(caps_crp, [], [ [0.5, 0.5] for _ in range(len(df)) ])
        cap_bh = 10000.0; caps_bh = [cap_bh]; w_bh = np.array([0.5, 0.5])
        for i in range(len(df)):
            R_t = np.array([df.iloc[i]['ret_spx'], df.iloc[i]['ret_cmc']])
            cap_bh *= (1.0 + np.sum(w_bh * R_t)); caps_bh.append(cap_bh)
            w_bh = w_bh * (1.0 + R_t); w_bh /= np.sum(w_bh) if np.sum(w_bh)>0 else 1.0
        met_bh = calc_metrics(caps_bh, [], [ [0.5, 0.5] for _ in range(len(df)) ])
        
        plt.figure(figsize=(12, 7))
        plt.plot(x_weeks, caps, label="Agente SAC (Base)", color='blue', linewidth=2.5)
        plt.plot(x_weeks, caps_crp, label="Rebalanceo Constante (50/50)", color='green', linestyle='--', linewidth=2)
        plt.plot(x_weeks, caps_bh, label="Naive B&H (50/50)", color='black', linestyle=':', linewidth=2)
        plt.title(f"Evolución del Capital - {p_name.upper()}", fontsize=16)
        plt.xlabel("Semanas"); plt.ylabel("Capital ($)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Capital_{p_name}.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.stackplot(x_weeks, w_spx, w_cmc, labels=['SPX', 'CMC200'], colors=['#1f77b4', '#ff7f0e'], alpha=0.8)
        plt.title(f"Estrategia de Inversión - {p_name.upper()}", fontsize=16)
        plt.xlabel("Semanas"); plt.ylim(0, 1)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Estrategia_Temporal_{p_name}.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(w_spx, bins=30, color='purple', alpha=0.7, edgecolor='black')
        plt.title(f"Histograma de Pesos SPX - {p_name.upper()}", fontsize=16)
        plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Histograma_Pesos_{p_name}.png"))
        plt.close()
        
        fig_q, ax_q1 = plt.subplots(figsize=(12, 6))
        color_q = 'blue'
        ax_q1.set_xlabel("Semanas")
        ax_q1.set_ylabel("Predicción Q(s,a) (El Crítico)", color=color_q, fontsize=12)
        line1 = ax_q1.plot(x_weeks[:-1], q_preds, label="Predicción Q(s,a)", color=color_q, linewidth=2)
        ax_q1.tick_params(axis='y', labelcolor=color_q)
        
        ax_q2 = ax_q1.twinx()  
        color_g = 'red'
        ax_q2.set_ylabel("Retorno Real Descontado $G_t$", color=color_g, fontsize=12)
        line2 = ax_q2.plot(x_weeks[:-1], G_t, label="Retorno Real Descontado $G_t$", color=color_g, alpha=0.6, linestyle='--')
        ax_q2.tick_params(axis='y', labelcolor=color_g)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_q1.legend(lines, labels, loc='best')
        
        plt.title(f"Crítico SAC: Correlación Q(s,a) vs Realidad - {p_name.upper()}", fontsize=16)
        ax_q1.grid(True, alpha=0.3)
        fig_q.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Value_Function_{p_name}.png"))
        plt.close(fig_q)
        
        if p_name in ['val', 'test']:
            if 'norm_var_mix_cmc' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df['norm_var_mix_cmc'].values, w_cmc[1:], alpha=0.7, color='teal', edgecolors='black', s=50)
                plt.title(f"Prueba de Cordura - {p_name.upper()}")
                z = np.polyfit(df['norm_var_mix_cmc'].values, w_cmc[1:], 1); p = np.poly1d(z)
                plt.plot(df['norm_var_mix_cmc'].values, p(df['norm_var_mix_cmc'].values), "r--", linewidth=2, label="Tendencia")
                plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"Scatter_Feature_{p_name}.png"))
                plt.close()
                
            window = min(12, max(4, len(df) // 4))
            returns = np.diff(caps) / caps[:-1]
            roll_sharpe = pd.Series(returns).rolling(window).apply(lambda x: (np.mean(x)/np.std(x))*np.sqrt(52) if np.std(x)>0 else 0)
            def calc_maxdd(x): return np.max((np.maximum.accumulate(x) - x) / np.maximum.accumulate(x)) * 100
            roll_maxdd = pd.Series(caps).rolling(window).apply(calc_maxdd)
            fig_r, ax_r1 = plt.subplots(figsize=(12, 6))
            ax_r1.plot(x_weeks[1:], roll_sharpe, color='green', label=f'Sharpe', linewidth=2)
            ax_r1.axhline(0, color='black', linestyle=':')
            ax_r2 = ax_r1.twinx()
            ax_r2.plot(x_weeks, roll_maxdd, color='red', label=f'Max Drawdown', linestyle='--', linewidth=2)
            plt.title(f"Rolling Metrics - {p_name.upper()}")
            lines_1, labels_1 = ax_r1.get_legend_handles_labels(); lines_2, labels_2 = ax_r2.get_legend_handles_labels()
            ax_r1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Rolling_Metrics_{p_name}.png"))
            plt.close(fig_r)

        print(f"{'Modelo':<25} | {'Cap. Final ($)':<14} | {'Recompensa':<10} | {'Sharpe':<8} | {'MaxDD(%)':<8} | {'Rotación(%)':<10}")
        print("-" * 90)
        print(f"{'SAC Base':<25} | ${met_sac[4]:<13.2f} | {met_sac[0]:<10.4f} | {met_sac[1]:<8.2f} | {met_sac[2]:<8.2f} | {met_sac[3]:<10.2f}")
        print(f"{'Rebalanceo Constante (CRP)':<25} | ${met_crp[4]:<13.2f} | {met_crp[0]:<10.4f} | {met_crp[1]:<8.2f} | {met_crp[2]:<8.2f} | {met_crp[3]:<10.2f}")
        print(f"{'Naive B&H (50/50)':<25} | ${met_bh[4]:<13.2f} | {met_bh[0]:<10.4f} | {met_bh[1]:<8.2f} | {met_bh[2]:<8.2f} | {met_bh[3]:<10.2f}")
        print("-" * 90)

if __name__ == "__main__":
    main()
