import numpy as np

# Cargar y mostrar datos
data = np.load('data/synthetic_data.npy', allow_pickle=True).item()

print("📊 RESUMEN DEL DATASET SINTÉTICO")
print(f"   Total de muestras: {len(data['states'])}")
print(f"   Forma de estados: {data['states'].shape}")
print(f"   Aceptación promedio: {data['acceptances'].mean():.2f}%")
print(f"   Aceptación máx: {data['acceptances'].max():.2f}%")
print(f"   Aceptación mín: {data['acceptances'].min():.2f}%")

print("\n🔍 PRIMERA MUESTRA (ejemplo):")
print(f"   Estado: {data['states'][0][:4]}...")  # Primeros 4 factores
print(f"   Acción: {data['actions'][0]}")        # Ajustes 4P
print(f"   Recompensa: {data['rewards'][0]:.3f}")
print(f"   Aceptación: {data['acceptances'][0]:.1f}%")