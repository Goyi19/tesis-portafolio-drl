import os
import sys
import numpy as np
import pandas as pd

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.sac_agent import SACPortfolioAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def run_experiment():
    handler = DataHandler()
    train_df = handler.get_partition('train')
    val_df = handler.get_partition('val')
    
    # Hiperparámetros exactos idénticos a los de PPO
    lambdas = [0.3, 0.7, 1.1]
    tc_levels = {
        "Friccion_Baja": [0.00001, 0.00004],
        "Friccion_Media": [0.00020, 0.00080],
        "Friccion_Alta": [0.00050, 0.00200]
    }
    
    base_output_dir = "models/sac_risk_cost_sens"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Iniciando Análisis de Sensibilidad (Risk vs Costs) - SAC")
    print(f"Total de experimentos: {len(lambdas) * len(tc_levels)}")
    print("-" * 60)
    
    for l_val in lambdas:
        for tc_name, tc_val in tc_levels.items():
            # ID único
            exp_id = f"Lambda_{str(l_val).replace('.', 'p')}_{tc_name}"
            exp_dir = os.path.join(base_output_dir, exp_id)
            
            # Verificación de existencia (reanudar si falla)
            if os.path.exists(os.path.join(exp_dir, "best_model.zip")):
                print(f"[SKIP] {exp_id} ya existe.")
                continue
                
            print(f"\n>>> EJECUTANDO SAC: {exp_id}")
            print(f"    Lambda: {l_val} | TC (SPX, CMC): {tc_val}")
            
            # Entornos
            env_train = PortfolioEnv(df=train_df, lambda_risk=l_val, transaction_cost_pct=tc_val)
            env_val_raw = PortfolioEnv(df=val_df, lambda_risk=l_val, transaction_cost_pct=tc_val, is_eval=True)
            env_val_monitored = Monitor(env_val_raw)
            env_val = DummyVecEnv([lambda: env_val_monitored])
            
            # Inicializar Agente SAC con arquitectura idéntica a PPO
            agent = SACPortfolioAgent(
                env=env_train, 
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=128,
                ent_coef="auto",
                seed=42, 
                verbose=0,
                tensorboard_log="logs/sac_sens_tensorboard/"
            )
            
            # Entrenamiento (57,000 pasos exactos)
            agent.train(
                total_timesteps=57000,
                eval_env=env_val,
                save_path=exp_dir,
                tb_log_name=f"SAC_{exp_id}"
            )
            
            print(f"[OK] Modelo guardado en {exp_dir}")

if __name__ == "__main__":
    run_experiment()
