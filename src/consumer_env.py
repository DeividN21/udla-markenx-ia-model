import gymnasium as gym
import numpy as np

class ConsumerEnv(gym.Env):
    def __init__(self):
        super(ConsumerEnv, self).__init__()
        # Espacio de observación: 4 factores + 4 ajustes 4P + 2 macro (ejemplo, N=10)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        # Espacio de acción: Ajustes en 4P (vector de 4 deltas)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.current_state = None
        self.reset()

    def reset(self, seed=None, options=None):
        # Inicializa estado sintético: factores aleatorios normalizados [0,1]
        self.current_state = np.random.uniform(0, 1, 10)
        return self.current_state, {}

    def step(self, action):
        # FÓRMULA CORREGIDA: Recompensas más realistas (20-90%)
        factor_match = np.sum(self.current_state[:4] * (action + 1) / 2)  # [0,2] → [0,1]
        cost_penalty = np.sum(np.abs(action)) * 0.05  # Penalización suave
        reward = factor_match - cost_penalty  # Rango: 0.2 a 0.9
        
        # Aceptación: 20-90%
        acceptance = np.clip(20 + reward * 70, 20, 90)
        
        done = False
        truncated = False
        return self.current_state, reward, done, truncated, {"acceptance": acceptance}

    def render(self):
        pass  # No visual por ahora