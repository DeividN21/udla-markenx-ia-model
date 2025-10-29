import torch
import torch.nn as nn
import numpy as np

class AcceptanceNN(nn.Module):
    def __init__(self, input_size=22):  # Tamaño input: one-hot de variables (~22 con one-hot)
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularización
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probabilidad [0,1]
        )

    def forward(self, x):
        return self.layers(x) * 100  # Porcentaje

# Función para codificar inputs
def encode_inputs(state, action):
    # Ejemplo simple: concatena state (10) + one-hot para categorías
    # Expande con one-hot para Country, etc.
    encoded = np.concatenate([state, action, np.zeros(14)])  # Placeholder para 22 features
    return torch.tensor(encoded, dtype=torch.float32)