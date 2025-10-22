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
        # Calcula recompensa: suma ponderada de coincidencias
        reward = np.sum(self.current_state[:4] * action) - np.abs(np.sum(action)) * 0.1  # Penaliza overuse
        # Nueva aceptación (simulada)
        acceptance = np.clip(np.sum(self.current_state) + reward, 0, 100)
        done = False  # No termina hasta fin de episodios
        truncated = False
        return self.current_state, reward, done, truncated, {"acceptance": acceptance}

    def render(self):
        pass  # No visual por ahora