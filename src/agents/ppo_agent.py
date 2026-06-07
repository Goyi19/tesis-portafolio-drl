import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class PPOPortfolioAgent:
    """
    Agente DRL basado en PPO (Proximal Policy Optimization).
    Sigue las especificaciones de Etapa 3 de la metodología.
    """
    def __init__(self, env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                 n_epochs=10, ent_coef=0.01, clip_range=0.2, vf_coef=0.5, seed=42, verbose=1,
                 tensorboard_log="logs/ppo_tensorboard/"):
        self.env = Monitor(env) # Envuelto en Monitor para registrar episodios
        self.seed = seed
        
        # Arquitectura de la red (MlpPolicy con 2 capas ocultas de 64 neuronas cada una)
        # Esto mitiga el sobreajuste como se discute en el documento.
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            clip_range=clip_range,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            seed=self.seed,
            verbose=verbose,
            device="auto",
            tensorboard_log=tensorboard_log
        )
        
    def train(self, total_timesteps, eval_env=None, save_path=None, tb_log_name="PPO"):
        """
        Entrena el agente PPO.
        Si se provee eval_env, utilizará EvalCallback para control de capacidad 
        (medición de J_val en validación).
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
            
        print(f"Iniciando entrenamiento de PPO por {total_timesteps} steps...")
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
        self.model = PPO.load(path, env=self.env)
