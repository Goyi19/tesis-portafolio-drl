import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    """
    Entorno MDP para optimización de portafolios (SPX, CMC200) vía DRL.
    Sigue rigurosamente las definiciones matemáticas de la tesis:
    - Acción: Vector continuo [-1, 1], proyectado al símplex.
    - Observación: [x_t, w_{t-1}, t/T, features_mix_normalizadas].
    - Recompensa: Media-Varianza con costos de transacción.
    - Dinámica: Actualización de capital con retornos reales y costos.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_capital=10000.0, transaction_cost_pct=[0.005, 0.010], lambda_risk=0.10, feature_cols=None, window_size=52, is_eval=False, noise_std=0.001):
        super(PortfolioEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.T = len(self.df)
        
        self.initial_capital = float(initial_capital)
        # Convertimos los costos de transacción a un array de numpy
        self.transaction_cost_pct = np.array(transaction_cost_pct, dtype=np.float32)
        self.lambda_risk = float(lambda_risk)
        self.window_size = window_size
        self.is_eval = is_eval
        self.noise_std = noise_std
        
        self.start_index = 0
        self.end_index = self.T
        
        if feature_cols is None:
            self.feature_cols = [
                'norm_mu_mix_spx', 
                'norm_mu_mix_cmc', 
                'norm_var_mix_spx', 
                'norm_var_mix_cmc', 
                'norm_cov_mix'
            ]
        else:
            self.feature_cols = feature_cols
            
        # Espacio de acción: a_t \in [-1, 1] para 2 activos (SPX y CMC200)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Espacio de observación:
        # [x_t, w_{SPX, t-1}, w_{CMC, t-1}, t/T, features...]
        # Dimensión = 1 (capital) + 2 (pesos) + 1 (tiempo) + len(features)
        obs_dim = 1 + 2 + 1 + len(self.feature_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Estado interno
        self.current_step = 0
        self.capital = self.initial_capital
        self.weights = np.array([0.5, 0.5], dtype=np.float32) # Pesos iniciales: equitativos
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.is_eval:
            # En evaluación (Valid/Test) queremos recorrer toda la serie sin cortes aleatorios
            self.start_index = 0
            self.end_index = self.T
        else:
            # En entrenamiento (Train) aplicamos Random Start para evitar memorización
            max_start = max(0, self.T - self.window_size)
            if max_start > 0:
                self.start_index = np.random.randint(0, max_start + 1)
            else:
                self.start_index = 0
            self.end_index = min(self.start_index + self.window_size, self.T)
            
        self.current_step = self.start_index
        self.capital = self.initial_capital
        self.weights = np.array([0.5, 0.5], dtype=np.float32)
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        # Evitamos out of bounds si current_step llega a T
        step = min(self.current_step, self.T - 1)
        row = self.df.iloc[step]
        
        features = row[self.feature_cols].values.astype(np.float32)
        
        # Tiempo relativo a la ventana (0.0 al inicio, ~1.0 al final). Escala consistente en Train/Val/Test
        steps_passed = step - self.start_index
        time_frac = np.array([steps_passed / self.window_size], dtype=np.float32)
        
        # Normalizamos el capital para la red neuronal (e.g. relativo al inicial)
        cap_arr = np.array([self.capital / self.initial_capital], dtype=np.float32)
        
        obs = np.concatenate([cap_arr, self.weights, time_frac, features])
        
        # Inyección de Ruido Gaussiano (Data Augmentation) solo en Entrenamiento
        if not self.is_eval and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=obs.shape)
            obs = obs + noise
            
        return obs.astype(np.float32)
        
    def step(self, action):
        # 1. Proyección Simplex
        a = np.clip(np.array(action).flatten(), -1.0, 1.0)
        a_clipped = np.maximum(0, a)
        sum_a = np.sum(a_clipped)
        if sum_a > 1e-8:
            w_t = a_clipped / sum_a
        else:
            w_t = np.array([0.5, 0.5], dtype=np.float32) # Fallback si ambas acciones son negativas
            
        # 2. Extraer datos de mercado del paso actual (t)
        row = self.df.iloc[self.current_step]
        
        # Retornos reales financieros (R_t) para evolución de capital
        R_t = np.array([row['ret_spx'], row['ret_cmc']])
        
        # Momentos esperados teóricos (no normalizados) para la Recompensa
        mu_spx = row['mu_mix_spx']
        mu_cmc = row['mu_mix_cmc']
        var_spx = row['var_mix_spx']
        var_cmc = row['var_mix_cmc']
        cov = row['cov_mix']
        
        # 3. Costos de Transacción proporcionales a la rotación (heterogéneos por activo)
        delta_w = w_t - self.weights
        c_t = np.sum(np.abs(delta_w) * self.transaction_cost_pct)
        
        # 4. Evolución del Capital Real
        # x_{t+1} = x_t * (1 + sum(w_i R_i) - c_t)
        port_ret_real = np.sum(w_t * R_t)
        self.capital = self.capital * (1.0 + port_ret_real - c_t)
        
        # 5. Recompensa (Reward) Media-Varianza
        # r_t = w^T \mu - \lambda (w^T \Sigma w) - c_t
        port_mu_exp = w_t[0] * mu_spx + w_t[1] * mu_cmc
        port_var_exp = (w_t[0]**2 * var_spx) + (w_t[1]**2 * var_cmc) + (2 * w_t[0] * w_t[1] * cov)
        
        reward = port_mu_exp - (self.lambda_risk * port_var_exp) - c_t
        
        # 6. Actualizar estado y verificar fin del episodio
        self.weights = w_t
        self.current_step += 1
        
        terminated = False
        if self.capital <= 0:
            self.capital = 0
            reward -= 10.0 # Penalización por ruina
            terminated = True
        elif self.current_step >= self.end_index:
            terminated = True
            
        # O_{t+1}
        obs = self._get_observation()
        
        # Diccionario de información adicional
        info = {
            'capital': self.capital,
            'port_ret_real': port_ret_real,
            'transaction_costs': c_t,
            'weights_spx': w_t[0],
            'weights_cmc': w_t[1],
            'reward_media_var': reward
        }
        
        truncated = False
        
        return obs, float(reward), terminated, truncated, info
