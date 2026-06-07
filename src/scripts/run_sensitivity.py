import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.ppo_agent import PPOPortfolioAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def train_variation(name, env_kwargs, agent_kwargs, steps=57000):
    print(f"\n=========================================")
    print(f"Iniciando Entrenamiento: {name}")
    print(f"=========================================")
    
    # Cargar Datos (Train y Val para ser idénticos al oficial)
    handler = DataHandler()
    train_df = handler.get_partition('train')
    val_df = handler.get_partition('val')
    
    # Configurar Entornos
    env_train = PortfolioEnv(df=train_df, **env_kwargs)
    env_val_raw = PortfolioEnv(df=val_df, is_eval=True, **env_kwargs)
    env_val_monitored = Monitor(env_val_raw)
    env_val = DummyVecEnv([lambda: env_val_monitored])
    
    # Rutas de guardado
    save_path = f"models/ppo_sens/{name}/best_model"
    tb_log = f"logs/ppo_sens_tensorboard"
    
    # Configurar Agente (Heredando el n_steps y batch_size calcado del oficial)
    agent = PPOPortfolioAgent(
        env=env_train,
        tensorboard_log=tb_log,
        verbose=0,
        **agent_kwargs
    )
    
    # Entrenar (Pasando eval_env para que guarde best_model)
    agent.train(total_timesteps=steps, eval_env=env_val, save_path=save_path, tb_log_name=name)
    # Sobrescribimos el nombre interno de tensorboard renombrando la carpeta generada
    # sb3 no nos deja setear el log_name en learn fácilmente si usamos wrapper
    print(f"Entrenamiento {name} finalizado y guardado en {save_path}")

def main():
    # 1. Definir los parámetros BASELINE
    base_env = {
        'initial_capital': 10000.0,
        'transaction_cost_pct': [0.005, 0.010],
        'lambda_risk': 0.10
    }
    
    base_agent = {
        'learning_rate': 3e-4,
        'n_epochs': 10,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'n_steps': 1140,      # Igual al oficial
        'batch_size': 114     # Igual al oficial
    }
    
    # 2. Definir las variaciones (Ceteris Paribus)
    experiments = {
        "Baseline": ({}, {}),
        
        # 1. Learning Rate
        "LR_Low": ({}, {'learning_rate': 5e-5}),
        "LR_High": ({}, {'learning_rate': 1e-3}),
        
        # 2. Entropy Coefficient
        "Ent_Low": ({}, {'ent_coef': 0.001}),
        "Ent_High": ({}, {'ent_coef': 0.05}),
        
        # 3. Clip Range
        "Clip_Low": ({}, {'clip_range': 0.1}),
        "Clip_High": ({}, {'clip_range': 0.3}),
        
        # 4. N Epochs
        "Epochs_Low": ({}, {'n_epochs': 5}),
        "Epochs_High": ({}, {'n_epochs': 20}),
        
        # 5. Risk Aversion (Entorno)
        "Risk_Low": ({'lambda_risk': 0.0}, {}),
        "Risk_High": ({'lambda_risk': 1.0}, {}),
        
        # 6. Transaction Costs (Entorno)
        "Cost_Zero": ({'transaction_cost_pct': [0.0, 0.0]}, {}),
        "Cost_High": ({'transaction_cost_pct': [0.02, 0.04]}, {})
    }
    
    # Limpiar logs anteriores de sensibilidad si existen
    if os.path.exists("logs/ppo_sens_tensorboard"):
        shutil.rmtree("logs/ppo_sens_tensorboard")
        
    # Ejecutar experimentos
    for exp_name, (env_updates, agent_updates) in experiments.items():
        # Copiar diccionarios base
        current_env = base_env.copy()
        current_agent = base_agent.copy()
        
        # Aplicar actualizaciones
        current_env.update(env_updates)
        current_agent.update(agent_updates)
        
        # Entrenar (usamos menos steps para que el análisis sea rápido, o los 57000 completos)
        # 57000 steps = 500 episodios
        train_variation(exp_name, current_env, current_agent, steps=57000)

if __name__ == "__main__":
    main()
