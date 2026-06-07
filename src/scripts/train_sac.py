import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.sac_agent import SACPortfolioAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

def main():
    # 1. Cargar datos y crear directorios
    print("Cargando datos y configurando directorios...")
    handler = DataHandler()
    train_df = handler.get_partition('train')
    val_df = handler.get_partition('val')
    
    os.makedirs('models/sac', exist_ok=True)
    
    # 2. Instanciar Entornos
    # Capital inicial 10,000, lambda = 0.10, costos = [0.5%, 1.0%]
    # Instanciar Entornos crudos
    env_train_raw = PortfolioEnv(df=train_df)
    env_val_raw = PortfolioEnv(df=val_df, is_eval=True)
    
    # Envolver en DummyVecEnv para compatibilidad nativa con SB3
    env_val_monitored = Monitor(env_val_raw)
    env_val = DummyVecEnv([lambda: env_val_monitored])
    
    # --- LOGICA DE ROLLOUT Y EPISODIOS ---
    episode_length = len(train_df)  # 114
    total_episodes = 500
    total_timesteps = episode_length * total_episodes # 114 * 500 = 57,000 pasos
    
    print(f"\n--- Configuración de Entrenamiento SAC ---")
    print(f"Longitud del episodio (Train): {episode_length} semanas")
    print(f"Total timesteps: {total_timesteps} ({total_episodes} episodios)\n")
    
    # 3. Inicializar Agente
    agent = SACPortfolioAgent(
        env=env_train_raw, # SACPortfolioAgent hace Monitor(env) internamente
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=128,
        ent_coef="auto",
        seed=42,
        verbose=1
    )
    
    # 4. Entrenar
    save_path = 'models/sac/best_model'
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=env_val,
        save_path=save_path
    )
    
    # Guardar modelo final
    agent.save('models/sac/final_model')
    print("Modelo final guardado en models/sac/final_model.zip")

if __name__ == "__main__":
    main()
