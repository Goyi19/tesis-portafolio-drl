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

def simulate_crp(df, tc_val=[0.0002, 0.0008]):
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

def plot_all():
    handler = DataHandler()
    parts = {'train': handler.get_partition('train'), 'val': handler.get_partition('val'), 'test': handler.get_partition('test')}
    
    lambdas = [0.3, 0.7, 1.1]
    tc_levels = {
        "Friccion_Baja": [0.00001, 0.00004],
        "Friccion_Media": [0.00020, 0.00080],
        "Friccion_Alta": [0.00050, 0.00200]
    }
    tc_keys = list(tc_levels.keys())
    
    models_dir = "models/sac_risk_cost_sens"
    out_dir = "results/figures/Analisis_Sensibilidad_SAC"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. GRÁFICOS DE RED
    fig_net, axes_net = plt.subplots(2, 2, figsize=(15, 10))
    fig_net.suptitle("Métricas de Red Neuronal SAC (Dual Sensitivity)", fontsize=16)
    titles = {
        'reward': 'Recompensa Media',
        'critic_loss': 'Critic Loss (Q-Network)',
        'actor_loss': 'Actor Loss (Policy)',
        'ent_coef': 'Coeficiente de Entropía (alpha)'
    }
    
    for p_name, df in parts.items():
        print(f"\nProcesando Partición: {p_name.upper()}")
        
        sim_results = {}
        if p_name == 'train': x_weeks = np.arange(len(df)+1)
        elif p_name == 'val': x_weeks = np.arange(len(parts['train']), len(parts['train'])+len(df)+1)
        else: x_weeks = np.arange(len(parts['train'])+len(parts['val']), len(parts['train'])+len(parts['val'])+len(df)+1)
        
        for l_val in lambdas:
            for tc_name, tc_val in tc_levels.items():
                exp_id = f"Lambda_{str(l_val).replace('.', 'p')}_{tc_name}"
                model_path = os.path.join(models_dir, exp_id, "best_model.zip")
                if not os.path.exists(model_path): continue
                
                env = PortfolioEnv(df=df, lambda_risk=l_val, transaction_cost_pct=tc_val, is_eval=True)
                agent = SACPortfolioAgent(env=env, verbose=0)
                agent.load(model_path[:-4])
                
                obs, _ = env.reset(); caps = [10000.0]; w_spx = [0.5]; w_cmc = [0.5]; rews = []; acts = []
                q_preds = []
                
                term = trunc = False
                while not (term or trunc):
                    # Predicción de Acción Determinista
                    action = agent.predict(obs, deterministic=True)
                    
                    # Predicción del Crítico Q(s,a)
                    obs_tensor, _ = agent.model.policy.obs_to_tensor(obs)
                    action_tensor = torch.tensor(action, dtype=torch.float32).reshape(1, -1).to(agent.model.device)
                    with torch.no_grad():
                        q_values = agent.model.critic(obs_tensor, action_tensor)
                        q_s_a = torch.min(q_values[0], q_values[1]).item()
                    q_preds.append(q_s_a)
                    
                    obs, r, term, trunc, info = env.step(action)
                    caps.append(info['capital']); w_spx.append(info['weights_spx']); w_cmc.append(info['weights_cmc'])
                    rews.append(r); acts.append(action)
                
                # Calcular Retorno Descontado G_t
                gamma = 0.99
                G_t = np.zeros(len(rews))
                g = 0
                for t in reversed(range(len(rews))):
                    g = rews[t] + gamma * g
                    G_t[t] = g
                    
                sim_results[exp_id] = {
                    'caps': caps, 'w_spx': w_spx, 'w_cmc': w_cmc, 
                    'q_preds': q_preds, 'G_t': G_t,
                    'metrics': calc_metrics(caps, rews, acts), 'l': l_val, 'tc': tc_val
                }
                
                if p_name == 'train':
                    tb_dirs = glob.glob(os.path.join("logs/sac_sens_tensorboard", f"SAC_{exp_id}*"))
                    if not tb_dirs:
                        all_logs = sorted(glob.glob(os.path.join("logs/sac_sens_tensorboard", "SAC_*")), key=os.path.getctime)
                        exp_idx = (lambdas.index(l_val) * 3) + list(tc_levels.keys()).index(tc_name)
                        if len(all_logs) > exp_idx + 1:
                            tb_dirs = [all_logs[exp_idx + 1]]
                    
                    if tb_dirs:
                        tb_data = extract_tb_metrics(tb_dirs[0])
                        for k, (steps, vals) in tb_data.items():
                            ax = axes_net[0 if k in ['reward', 'critic_loss'] else 1, 0 if k in ['reward', 'actor_loss'] else 1]
                            ax.plot(steps, vals, label=f"L={l_val}, TC={tc_val}", alpha=0.7)
                            ax.set_title(titles[k])

        if not sim_results: continue
        
        # Identificar el Mejor Modelo de esta partición
        best_eid = max(sim_results.keys(), key=lambda eid: sim_results[eid]['metrics'][4])
        best_data = sim_results[best_eid]

        # --- FIGURA CAPITAL ---
        plt.figure(figsize=(14, 8))
        for eid, data in sim_results.items():
            plt.plot(x_weeks, data['caps'], label=f"λ={data['l']}, TC={data['tc']}")
        
        plt.plot(x_weeks, simulate_crp(df), label="Rebalanceo Constante (50/50)", color='green', linestyle='--', linewidth=2)
        
        cap_bh = 10000.0; caps_bh = [cap_bh]; w_bh = np.array([0.5, 0.5])
        for i in range(len(df)):
            R_t = np.array([df.iloc[i]['ret_spx'], df.iloc[i]['ret_cmc']])
            cap_bh *= (1.0 + np.sum(w_bh * R_t)); caps_bh.append(cap_bh)
            w_bh = w_bh * (1.0 + R_t); w_bh /= np.sum(w_bh) if np.sum(w_bh)>0 else 1.0
        plt.plot(x_weeks, caps_bh, label="Naive B&H (50/50)", color='black', linestyle=':', linewidth=3)
        
        plt.title(f"Evolución del Capital SAC - Set de {p_name.upper()}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Capital_Dual_{p_name}.png"))
        plt.close()

        # --- SUPER GRÁFICO 3x3 ---
        fig_q, axes_q = plt.subplots(3, 3, figsize=(20, 15), sharex=True, sharey=True)
        fig_q.suptitle(f"Estrategia de Inversión SAC (Pesos) - Set de {p_name.upper()}", fontsize=20)
        for i, l_val in enumerate(lambdas):
            for j, tcn in enumerate(tc_keys):
                eid = f"Lambda_{str(l_val).replace('.', 'p')}_{tcn}"
                ax = axes_q[i, j]
                if eid in sim_results:
                    d = sim_results[eid]
                    ax.stackplot(x_weeks, d['w_spx'], d['w_cmc'], labels=['SPX (S&P 500)', 'CMC200 (Cripto)'], colors=['#1f77b4', '#ff7f0e'], alpha=0.8)
                    ax.set_title(f"λ={l_val} | TC={d['tc']}", fontsize=11); ax.set_ylim(0, 1)
                
                if i == 2: ax.set_xlabel("Semanas", fontsize=12)
                if j == 0: ax.set_ylabel("Distribución de Pesos", fontsize=12)
        
        handles, labels = axes_q[0, 0].get_legend_handles_labels()
        if handles: fig_q.legend(handles, labels, loc='upper right', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"SuperGrafico_Estrategia_Dual_{p_name}.png"))
        plt.close(fig_q)

        # --- HISTOGRAMA 3x3 ---
        fig_h, axes_h = plt.subplots(3, 3, figsize=(20, 15), sharex=True, sharey=True)
        fig_h.suptitle(f"Histograma de Pesos SPX SAC - {p_name.upper()}", fontsize=20)
        for i, l_val in enumerate(lambdas):
            for j, tcn in enumerate(tc_keys):
                eid = f"Lambda_{str(l_val).replace('.', 'p')}_{tcn}"
                ax = axes_h[i, j]
                if eid in sim_results:
                    d = sim_results[eid]
                    ax.hist(d['w_spx'], bins=20, color='purple', alpha=0.7, edgecolor='black')
                    ax.set_title(f"λ={l_val} | TC={d['tc']}", fontsize=11)
                if i == 2: ax.set_xlabel("Peso Asignado a SPX", fontsize=12)
                if j == 0: ax.set_ylabel("Frecuencia (Semanas)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"Histograma_Pesos_{p_name}.png"))
        plt.close(fig_h)
        
        # --- PRECISIÓN Q(s,a) CON TWINX ---
        fig_v, ax_v1 = plt.subplots(figsize=(12, 6))
        color_q = 'blue'
        ax_v1.set_xlabel("Semanas")
        ax_v1.set_ylabel("Predicción Q(s,a) (El Crítico)", color=color_q, fontsize=12)
        line1 = ax_v1.plot(x_weeks[:-1], best_data['q_preds'], label="Predicción Q(s,a)", color=color_q, linewidth=2)
        ax_v1.tick_params(axis='y', labelcolor=color_q)
        
        ax_v2 = ax_v1.twinx()  
        color_g = 'red'
        ax_v2.set_ylabel("Retorno Real Descontado $G_t$", color=color_g, fontsize=12)
        line2 = ax_v2.plot(x_weeks[:-1], best_data['G_t'], label="Retorno Real Descontado $G_t$", color=color_g, alpha=0.6, linestyle='--')
        ax_v2.tick_params(axis='y', labelcolor=color_g)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_v1.legend(lines, labels, loc='best')
        
        plt.title(f"Crítico SAC: Correlación Q(s,a) vs Realidad (Mejor: {best_eid}) - {p_name.upper()}", fontsize=16)
        ax_v1.grid(True, alpha=0.3)
        fig_v.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Value_Function_{p_name}.png"))
        plt.close(fig_v)

        # Exclusivos Val y Test
        if p_name in ['val', 'test']:
            if 'norm_var_mix_cmc' in df.columns:
                feature_vals = df['norm_var_mix_cmc'].values
                w_cmc_decisions = best_data['w_cmc'][1:]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(feature_vals, w_cmc_decisions, alpha=0.6, color='teal', edgecolors='black')
                plt.xlabel("Volatilidad Cripto Predicha (norm_var_mix_cmc)")
                plt.ylabel("Peso de Inversión Asignado a CMC200")
                plt.title(f"Prueba de Cordura SAC (Mejor Modelo: {best_eid}) - {p_name.upper()}")
                
                z = np.polyfit(feature_vals, w_cmc_decisions, 1)
                p = np.poly1d(z)
                plt.plot(feature_vals, p(feature_vals), "r--", alpha=0.8, label="Tendencia")
                
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(out_dir, f"Scatter_Feature_{p_name}.png"))
                plt.close()
            
            window = min(12, max(4, len(df) // 4))
            returns = np.diff(best_data['caps']) / best_data['caps'][:-1]
            roll_sharpe = pd.Series(returns).rolling(window).apply(lambda x: (np.mean(x)/np.std(x))*np.sqrt(52) if np.std(x)>0 else 0)
            
            def calc_maxdd(x):
                r_max = np.maximum.accumulate(x)
                return np.max((r_max - x) / r_max) * 100
            
            roll_maxdd = pd.Series(best_data['caps']).rolling(window).apply(calc_maxdd)
            
            fig_r, ax_r1 = plt.subplots(figsize=(12, 6))
            ax_r1.plot(x_weeks[1:], roll_sharpe, color='green', label=f'Sharpe Ratio (Win={window})', linewidth=2)
            ax_r1.set_ylabel("Sharpe Ratio", color='green')
            ax_r1.axhline(0, color='black', linewidth=0.8, linestyle=':')
            
            ax_r2 = ax_r1.twinx()
            ax_r2.plot(x_weeks, roll_maxdd, color='red', label=f'Max Drawdown % (Win={window})', linestyle='--', linewidth=2)
            ax_r2.set_ylabel("Max Drawdown (%)", color='red')
            
            plt.title(f"Rolling Metrics de Estabilidad SAC (Mejor: {best_eid}) - {p_name.upper()}")
            lines_1, labels_1 = ax_r1.get_legend_handles_labels()
            lines_2, labels_2 = ax_r2.get_legend_handles_labels()
            ax_r1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
            
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Rolling_Metrics_{p_name}.png"))
            plt.close(fig_r)

        # --- TABLA DE MÉTRICAS ---
        metrics_bh = calc_metrics(caps_bh, [], [ [0.5, 0.5] for _ in range(len(df)) ])
        caps_crp = simulate_crp(df)
        metrics_crp = calc_metrics(caps_crp, [], [ [0.5, 0.5] for _ in range(len(df)) ])

        print(f"\nMETRICAS SAC {p_name.upper()}")
        print(f"{'Modelo':<30} | {'Cap. Final ($)':<15} | {'Recompensa':<10} | {'Sharpe':<8} | {'MaxDD(%)':<8} | {'Rotación(%)':<10}")
        print("-" * 100)
        
        all_results = []
        for eid, data in sim_results.items():
            all_results.append((eid, data['metrics']))
        all_results.append(("Naive B&H (50/50)", metrics_bh))
        all_results.append(("Rebalanceo Constante (CRP)", metrics_crp))
        
        for name, m in sorted(all_results, key=lambda x: x[1][4], reverse=True):
            print(f"{name:<30} | ${m[4]:<14.2f} | {m[0]:<10.4f} | {m[1]:<8.2f} | {m[2]:<8.2f} | {m[3]:<10.2f}")
        print("-" * 100)

    # Finalizar fig_net
    axes_net[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    fig_net.tight_layout(); fig_net.savefig(os.path.join(out_dir, "Metricas_Red_Dual.png"))
    plt.close(fig_net)

if __name__ == "__main__":
    plot_all()
