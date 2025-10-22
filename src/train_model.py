import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from consumer_env import ConsumerEnv


def generate_synthetic_data(num_samples=10000):
    """Genera dataset sintético para validación"""
    states = np.random.uniform(0, 1, (num_samples, 10))
    actions = np.random.uniform(-1, 1, (num_samples, 4))
    rewards = np.sum(states[:, :4] * actions, axis=1) - 0.1 * np.sum(np.abs(actions), axis=1)
    acceptances = np.clip(np.sum(states, axis=1) + rewards, 0, 100)
    data = {'states': states, 'actions': actions, 'rewards': rewards, 'acceptances': acceptances}
    np.save('data/synthetic_data.npy', data)
    print(f"✅ Dataset sintético generado: {num_samples} muestras guardadas en data/synthetic_data.npy")

def train_ppo():
    """Entrena el modelo PPO"""
    print("Iniciando entrenamiento PPO...")
    env = DummyVecEnv([lambda: ConsumerEnv()])
    
    # Red neuronal: 4 capas ocultas [256, 128, 128, 64]
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        policy_kwargs={'net_arch': [256, 128, 128, 64]},
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64
    )
    
    # Entrenamiento: 100k timesteps (rápido para demo)
    model.learn(total_timesteps=100000)
    
    # Guardar modelo
    model_path = "models/ppo_consumer_model"
    model.save(model_path)
    print(f"Modelo entrenado y guardado: {model_path}.zip")

def test_inference():
    """Prueba rápida de inferencia"""
    print("Probando inferencia...")
    model = PPO.load("models/ppo_consumer_model")
    obs = np.random.uniform(0, 1, 10)
    action, _ = model.predict(obs)
    print(f"   Estado de prueba: {obs[:4]}...")
    print(f"   Acción predicha (ajustes 4P): {action}")
    print("    Inferencia funcionando correctamente")

if __name__ == "__main__":
    print("=== MARKENX - ENTRENAMIENTO PPO ===\n")
    generate_synthetic_data()
    train_ppo()
    test_inference()