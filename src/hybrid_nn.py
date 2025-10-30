import torch
import torch.nn as nn
import numpy as np

class AcceptanceNN(nn.Module):
    def __init__(self, input_size=28):  # Cambiado a 28 para coincidir con encode (10 state + 4 action + 14 placeholders)
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x) * 100

# Función para codificar inputs
def encode_inputs(state, action):
    # Placeholder para one-hot: 14 dims (ajusta si expandes categorías del documento)
    placeholders = np.zeros(14)  # e.g., para Country(2), AgeGroup(2), etc.
    encoded = np.concatenate([state, action, placeholders])
    print(f"DEBUG: Encoded shape: {encoded.shape}")  # Depura
    return torch.tensor(encoded, dtype=torch.float32)