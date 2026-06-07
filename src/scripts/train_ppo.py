import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
from src.agents.ppo_agent import PPOPortfolioAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

def main():
    # 1. Cargar datos y crear directorios
    print("Cargando datos y configurando directorios...")
    handler = DataHandler()
    train_df = handler.get_partition('train')
    val_df = handler.get_partition('val')
    
    os.makedirs('models/ppo', exist_ok=True)
    
    # 2. Instanciar Entornos
    # Capital inicial 10,000, lambda = 0.10, costos = [0.5%, 1.0%]
    # Instanciar Entornos crudos
    env_train_raw = PortfolioEnv(df=train_df)
    env_val_raw = PortfolioEnv(df=val_df, is_eval=True)
    
    # Envolver en DummyVecEnv para compatibilidad nativa con SB3
    # Tambien envolvemos el eval_env en Monitor aquí (PPOPortfolioAgent ya envuelve su propio env en init, 
    # pero para evitar warnings y conflictos, mejor pasamos todo vectorizado).
    # OJO: PPOPortfolioAgent internamente envuelve self.env = Monitor(env).
    # Dado que pasaremos un DummyVecEnv, debemos remover o ajustar el Monitor interno.
    # En PPOPortfolioAgent, mantuvimos self.env = Monitor(env). Esto fallará si env ya es un VecEnv.
    # Así que pasaremos los raw envs a PPOPortfolioAgent, excepto eval_env que debe ir envuelto si es VecEnv.
    # La mejor práctica en SB3 es pasar todo como DummyVecEnv si queremos evitar warnings.
    env_val_monitored = Monitor(env_val_raw)
    env_val = DummyVecEnv([lambda: env_val_monitored])
    
    # --- LOGICA DE ROLLOUT Y EPISODIOS ---
    # La longitud de un episodio de entrenamiento es exactamente el número de semanas en train_df
    episode_length = len(train_df)  # 114
    
    # Definimos que PPO actualice sus pesos cada vez que termine exactamente 10 episodios completos.
    # Esto evita el "truncamiento" a mitad de un episodio.
    episodes_per_rollout = 10
    n_steps = episode_length * episodes_per_rollout  # 1140
    
    # El batch size debe ser un divisor exacto de n_steps. 
    # 1140 / 10 = 114. Cada minibatch procesará exactamente el equivalente a 1 episodio.
    batch_size = episode_length 
    
    # Queremos entrenar por un total de 500 episodios
    total_episodes = 500
    total_timesteps = episode_length * total_episodes # 114 * 500 = 57,000 pasos
    
    print(f"\n--- Configuración de Entrenamiento ---")
    print(f"Longitud del episodio (Train): {episode_length} semanas")
    print(f"n_steps (Rollout buffer): {n_steps} pasos ({episodes_per_rollout} episodios por actualización)")
    print(f"batch_size: {batch_size}")
    print(f"Total timesteps: {total_timesteps} ({total_episodes} episodios)\n")
    
    # 3. Inicializar Agente
    agent = PPOPortfolioAgent(
        env=env_train_raw, # PPOPortfolioAgent hace Monitor(env) internamente
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        seed=42,
        verbose=1
    )
    
    # 4. Entrenar
    save_path = 'models/ppo/best_model'
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=env_val,
        save_path=save_path
    )
    
    # Guardar modelo final (además del mejor guardado por validación)
    agent.save('models/ppo/final_model')
    print("Modelo final guardado en models/ppo/final_model.zip")

if __name__ == "__main__":
    main()
