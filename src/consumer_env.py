import gymnasium as gym
import numpy as np
import pgmpy.models as pgm
import pgmpy.factors.discrete as pgf
from pgmpy.inference import VariableElimination

class ConsumerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Espacio de observación: 10 features (adaptado a variables del documento)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        # Espacio de acción: 4 ajustes 4P
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.current_state = None
        self.bn = self._build_bn()  # Construye la BN
        self.reset()

    def _build_bn(self):
        # Estructura BN según documento (con correcciones: agrega Psychological como latente)
        bn = pgm.BayesianNetwork([
            ('Country', 'CulturalPreference'),
            ('Lifestyle', 'CulturalPreference'),
            ('IncomeLevel', 'PriceSensitivity'),
            ('AgeGroup', 'InnovationAffinity'),
            ('AgeGroup', 'SocialInfluence'),
            ('Lifestyle', 'SocialInfluence'),
            ('AgeGroup', 'PsychologicalFactors'),  # Agregado para factores psicológicos
            ('Gender', 'PsychologicalFactors'),
            ('CulturalPreference', 'Acceptance'),
            ('PriceSensitivity', 'Acceptance'),
            ('InnovationAffinity', 'Acceptance'),
            ('SocialInfluence', 'Acceptance'),
            ('PsychologicalFactors', 'Acceptance'),
            ('IsEconomic', 'Acceptance'),
            ('IsInnovative', 'Acceptance'),
            ('IsCulturallyAcceptable', 'Acceptance'),
            ('Season', 'Acceptance'),
            ('Availability', 'Acceptance'),
            ('PriceTier', 'Acceptance')
        ])

        # CPTs de ejemplo (simplificadas, como en sección 6; usa valores realistas)
        # Para Acceptance: binario (0=No, 1=Yes), condicionado a 2 padres para simplicidad
        cpt_acceptance = pgf.TabularCPD(
            variable='Acceptance', variable_card=2,
            values=[[0.15, 0.35, 0.6, 0.7, 0.85, 0.95], [0.85, 0.65, 0.4, 0.3, 0.15, 0.05]],
            evidence=['IsEconomic', 'PriceSensitivity'], evidence_card=[2, 3],
            state_names={'Acceptance': ['No', 'Yes'], 'IsEconomic': ['No', 'Yes'], 'PriceSensitivity': ['Low', 'Mid', 'High']}
        )

        # Agrega CPTs para nodos latentes (ejemplo para CulturalPreference)
        cpt_cultural = pgf.TabularCPD(
            variable='CulturalPreference', variable_card=3,
            values=[[0.6, 0.3, 0.1, 0.2, 0.5, 0.3], [0.3, 0.4, 0.5, 0.4, 0.3, 0.3], [0.1, 0.3, 0.4, 0.4, 0.2, 0.4]],
            evidence=['Country', 'Lifestyle'], evidence_card=[2, 3],
            state_names={'CulturalPreference': ['HighTradition', 'Neutral', 'Progressive'], 
                         'Country': ['Ecuador', 'Rusia'], 'Lifestyle': ['Traditional', 'Modern', 'TechSavvy']}
        )

        # Agrega CPTs para otros nodos (similares; expande según necesitas)
        bn.add_cpds(cpt_acceptance, cpt_cultural) 
        bn.check_model()  # Verifica validez
        return bn

    def reset(self, seed=None, options=None):
        self.current_state = np.random.uniform(0, 1, 10)  # Estados sintéticos
        return self.current_state, {}

    def step(self, action):
        # Mapear acción a evidencia (e.g., action[0] -> IsEconomic)
        evidence = {
            'IsEconomic': 1 if action[0] > 0 else 0,  # Yes/No
            'PriceSensitivity': np.random.choice([0, 1, 2])  # Low/Mid/High, simulado
            # Agrega más basados en state/action y tu DB
        }

        # Inferencia con BN
        infer = VariableElimination(self.bn)
        q = infer.query(variables=['Acceptance'], evidence=evidence)
        acceptance = q.values[1] * 100  # Prob de 'Yes' como porcentaje
        reward = acceptance / 100

        done = False
        truncated = False
        return self.current_state, reward, done, truncated, {"acceptance": acceptance}