import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class SACPortfolioAgent:
    """
    Agente DRL basado en SAC (Soft Actor-Critic).
    Adaptado arquitectónicamente para ser comparable de manera justa contra PPO.
    """
    def __init__(self, env, learning_rate=3e-4, buffer_size=100000, learning_starts=1000, 
                 batch_size=128, ent_coef="auto", seed=42, verbose=1,
                 tensorboard_log="logs/sac_tensorboard/"):
        self.env = Monitor(env) # Envuelto en Monitor para registrar episodios
        self.seed = seed
        
        # Arquitectura de la red (MlpPolicy con 2 capas ocultas de 64 neuronas cada una)
        # Idéntica a la utilizada en PPO para garantizar evaluación "apples-to-apples".
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], qf=[64, 64])
        )
        
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            seed=self.seed,
            verbose=verbose,
            device="auto",
            tensorboard_log=tensorboard_log
        )
        
    def train(self, total_timesteps, eval_env=None, save_path=None, tb_log_name="SAC"):
        """
        Entrena el agente SAC.
        Si se provee eval_env, utilizará EvalCallback para control de capacidad.
        """
        callbacks = []
        if eval_env is not None:
            # Evaluar periódicamente para evitar sobreajuste
            eval_callback = EvalCallback(
                eval_env, 
                best_model_save_path=save_path,
                log_path=save_path, 
                eval_freq=max(1000, total_timesteps // 20),
                deterministic=True, 
                render=False
            )
            callbacks.append(eval_callback)
            
        print(f"Iniciando entrenamiento de SAC por {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=tb_log_name)
        print("Entrenamiento finalizado.")
        
    def predict(self, obs, deterministic=True):
        """Predice la acción dado un vector de observación."""
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action
        
    def save(self, path):
        """Guarda el modelo entrenado."""
        self.model.save(path)
        
    def load(self, path):
        """Carga un modelo entrenado."""
        self.model = SAC.load(path, env=self.env)
