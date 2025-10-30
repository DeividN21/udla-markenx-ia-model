import gymnasium as gym
import numpy as np
import pgmpy.models as pgm
import pgmpy.factors.discrete as pgf
from pgmpy.inference import VariableElimination
from hybrid_nn import AcceptanceNN, encode_inputs  # Importa NN para Acceptance
import torch

class ConsumerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.current_state = None
        self.bn = self._build_bn()  # Construye BN para latentes
        self.nn_model = self._load_nn_model()  # Carga NN preentrenada para Acceptance
        self.reset()

    def _build_bn(self):
        # BN simplificada para latentes
        bn = pgm.DiscreteBayesianNetwork([
            ('Country', 'CulturalPreference'),
            ('Lifestyle', 'CulturalPreference'),
            ('IncomeLevel', 'PriceSensitivity'),
            ('AgeGroup', 'InnovationAffinity'),
            ('AgeGroup', 'SocialInfluence'),
            ('Lifestyle', 'SocialInfluence'),
            ('AgeGroup', 'PsychologicalFactors'),
            ('Gender', 'PsychologicalFactors')
        ])

        # CPDs para nodos raíz y latentes (completo, sin Acceptance)
        cpt_country = pgf.TabularCPD('Country', 2, [[0.5], [0.5]], state_names={'Country': ['Ecuador', 'Rusia']})
        cpt_lifestyle = pgf.TabularCPD('Lifestyle', 3, [[0.3], [0.4], [0.3]], state_names={'Lifestyle': ['Traditional', 'Modern', 'TechSavvy']})
        cpt_incomelevel = pgf.TabularCPD('IncomeLevel', 3, [[0.4], [0.4], [0.2]], state_names={'IncomeLevel': ['Low', 'Mid', 'High']})
        cpt_agegroup = pgf.TabularCPD('AgeGroup', 2, [[0.4], [0.6]], state_names={'AgeGroup': ['Minor', 'Adult']})
        cpt_gender = pgf.TabularCPD('Gender', 2, [[0.5], [0.5]], state_names={'Gender': ['Male', 'Female']})

        cpt_cultural = pgf.TabularCPD(
            'CulturalPreference', 3,
            [[0.6, 0.3, 0.1, 0.2, 0.5, 0.3], [0.3, 0.4, 0.5, 0.4, 0.3, 0.3], [0.1, 0.3, 0.4, 0.4, 0.2, 0.4]],
            evidence=['Country', 'Lifestyle'], evidence_card=[2, 3],
            state_names={'CulturalPreference': ['HighTradition', 'Neutral', 'Progressive'], 'Country': ['Ecuador', 'Rusia'], 'Lifestyle': ['Traditional', 'Modern', 'TechSavvy']}
        )

        cpt_pricesensitivity = pgf.TabularCPD(
            'PriceSensitivity', 3,
            [[0.2, 0.4, 0.6], [0.3, 0.4, 0.3], [0.5, 0.2, 0.1]],
            evidence=['IncomeLevel'], evidence_card=[3],
            state_names={'PriceSensitivity': ['Low', 'Mid', 'High'], 'IncomeLevel': ['Low', 'Mid', 'High']}
        )

        cpt_innovationaffinity = pgf.TabularCPD(
            'InnovationAffinity', 3,
            [[0.6, 0.2], [0.3, 0.4], [0.1, 0.4]],
            evidence=['AgeGroup'], evidence_card=[2],
            state_names={'InnovationAffinity': ['Low', 'Mid', 'High'], 'AgeGroup': ['Minor', 'Adult']}
        )

        cpt_socialinfluence = pgf.TabularCPD(
            'SocialInfluence', 3,
            [[0.4, 0.3, 0.2, 0.5, 0.4, 0.3], [0.4, 0.4, 0.4, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.2, 0.3, 0.4]],
            evidence=['AgeGroup', 'Lifestyle'], evidence_card=[2, 3],
            state_names={'SocialInfluence': ['Low', 'Mid', 'High'], 'AgeGroup': ['Minor', 'Adult'], 'Lifestyle': ['Traditional', 'Modern', 'TechSavvy']}
        )

        cpt_psychological = pgf.TabularCPD(
            'PsychologicalFactors', 3,
            [[0.4, 0.3, 0.5, 0.4], [0.4, 0.4, 0.3, 0.3], [0.2, 0.3, 0.2, 0.3]],
            evidence=['AgeGroup', 'Gender'], evidence_card=[2, 2],
            state_names={'PsychologicalFactors': ['Low', 'Mid', 'High'], 'AgeGroup': ['Minor', 'Adult'], 'Gender': ['Male', 'Female']}
        )

        # Agrega CPDs (sin Acceptance)
        bn.add_cpds(cpt_country, cpt_lifestyle, cpt_incomelevel, cpt_agegroup, cpt_gender,
                    cpt_cultural, cpt_pricesensitivity, cpt_innovationaffinity,
                    cpt_socialinfluence, cpt_psychological)
        
        bn.check_model()  # Ahora pasa
        return bn

    def _load_nn_model(self):
        model = AcceptanceNN(input_size=28)
        model.load_state_dict(torch.load('models/hybrid_nn_model.pth'))
        model.eval()  # Modo inferencia
        return model

    def reset(self, seed=None, options=None):
        self.current_state = np.random.uniform(0, 1, 10)
        return self.current_state, {}

    def step(self, action):
        # Infer latentes con BN (ejemplo simple)
        evidence = {'Country': 'Ecuador' if self.current_state[0] > 0.5 else 'Rusia',  # Simulado de state
                    'Lifestyle': 'Traditional' if self.current_state[1] < 0.3 else 'Modern' if self.current_state[1] < 0.7 else 'TechSavvy'}
        infer = VariableElimination(self.bn)
        q_cultural = infer.query(variables=['CulturalPreference'], evidence=evidence)
        # ... (infer otros latentes similarmente; para simplicidad, usa valores medios)

        # Prepara input para NN (todos los padres de Acceptance)
        nn_input = encode_inputs(self.current_state, action)  # State + action + placeholders para latentes/producto
        acceptance = self.nn_model(nn_input).item()  # Usar NN para P(Acceptance)

        reward = acceptance / 100
        done = False
        truncated = False
        return self.current_state, reward, done, truncated, {"acceptance": acceptance}