# MarkenX - Modelo de IA

## Descripción
Este repositorio contiene el backend de IA para el videojuego educativo MarkenX.
Utiliza Reinforcement Learning (PPO con PyTorch) para simular la aceptación del consumidor basada en factores culturales, sociales, personales y psicológicos, ajustes en las 4P (Producto, Precio, Plaza, Promoción) y eventos macroentorno.

## Requisitos
- Python 3.12+
- Bibliotecas: torch, stable-baselines3, gymnasium, numpy
- Instalación: `pip install -r requirements.txt` (crea un requirements.txt con `pip freeze > requirements.txt` después de instalar).

## Estructura de Carpetas
- `/src`: Código fuente (p. ej. entornos RL, scripts de entrenamiento).
- `/data`: Datasets sintéticos.
- `/models`: Modelos entrenados.

## ¿Cómo Ejecutar?
1. Clonar el repo: `git clone https://github.com/DeividN21/udla-markenx-ia-model.git`
2. Instala dependencias: `pip install torch stable-baselines3 gymnasium numpy`
3. Ejecuta entrenamiento: `python src/train_model.py`

## Integración
Este modelo se integra con el repositorio principal del videojuego: [udla-markenx-videogame](https://github.com/DeividN21/udla-markenx-videogame.git).
