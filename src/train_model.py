import os
import numpy as np
#from stable_baselines3 import PPO
#from stable_baselines3.common.vec_env import DummyVecEnv
from consumer_env import ConsumerEnv
from hybrid_nn import AcceptanceNN, encode_inputs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader  # Nuevo para batches

# Crear directorios
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def generate_synthetic_data(num_samples=10000):
    states = np.random.uniform(0, 1, (num_samples, 10))
    actions = np.random.uniform(-1, 1, (num_samples, 4))
    factor_match = np.sum(states[:, :4] * (actions + 1) / 2, axis=1)
    cost_penalty = np.sum(np.abs(actions), axis=1) * 0.05
    rewards = factor_match - cost_penalty
    acceptances = np.clip(20 + rewards * 70, 20, 90)
    data = {'states': states, 'actions': actions, 'rewards': rewards, 'acceptances': acceptances}
    file_path = 'data/synthetic_data_hybrid.npy'
    np.save(file_path, data)
    print(f"   Dataset híbrido generado: {num_samples} muestras guardadas en {file_path}")
    print(f"   Aceptación promedio: {acceptances.mean():.2f}%")

def train_hybrid_nn(batch_size=64):
    file_path = 'data/synthetic_data_hybrid.npy'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    data = np.load(file_path, allow_pickle=True).item()
    X_list = [encode_inputs(s, a) for s, a in zip(data['states'], data['actions'])]
    X = torch.stack(X_list)  # Stack a tensor (N x 28)
    y = torch.tensor(data['acceptances'] / 100, dtype=torch.float32).unsqueeze(1)  # N x 1
    
    print(f"DEBUG: Dataset cargado - Shape X: {X.shape}, Shape y: {y.shape}")
    
    # DataLoader para batches
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AcceptanceNN(input_size=28)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        total_loss = 0
        num_batches = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X) / 100  # Normaliza para BCE
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss {avg_loss:.4f} (Procesados {num_batches} batches)")
    
    torch.save(model.state_dict(), 'models/hybrid_nn_model.pth')
    print(" NN híbrida entrenada y guardada")

def test_inference():
    print(" Probando inferencia...")
    env = ConsumerEnv()
    obs = np.random.uniform(0, 1, 10)
    action = np.random.uniform(-1, 1, 4)
    _, reward, _, _, info = env.step(action)
    print(f"   Estado de prueba: {obs[:4]}...")
    print(f"   Acción (ajustes 4P): {action}")
    print(f"   Recompensa: {reward:.3f}")
    print(f"   Aceptación BN: {info['acceptance']:.1f}%")
    print(" Inferencia OK")

if __name__ == "__main__":
    print("=== MARKENX - ENTRENAMIENTO HÍBRIDO BN+NN ===\n")
    generate_synthetic_data()
    train_hybrid_nn()
    test_inference()

'''
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
'''