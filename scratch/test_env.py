import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_handler import DataHandler
from src.environment.portfolio_env import PortfolioEnv
import numpy as np

def test_environment():
    print("Cargando DataHandler...")
    handler = DataHandler()
    train_df = handler.get_partition('train')
    
    print("Inicializando PortfolioEnv...")
    env = PortfolioEnv(df=train_df, initial_capital=10000.0, transaction_cost_pct=[0.005, 0.010], lambda_risk=0.10)
    
    obs, info = env.reset()
    print(f"Observación inicial (Dim: {len(obs)}): {obs}")
    
    terminated = False
    truncated = False
    
    rewards = []
    capitals = []
    
    while not (terminated or truncated):
        # Tomar accion aleatoria entre -1 y 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        capitals.append(info['capital'])
        
    print(f"\nEpisodio finalizado.")
    print(f"Pasos simulados: {env.current_step}")
    print(f"Capital final: {capitals[-1]:.2f}")
    print(f"Recompensa acumulada: {sum(rewards):.4f}")
    print("El entorno funciona correctamente.")

if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"Error: {e}")
