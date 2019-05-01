from .q_learner import QLearner
from .coma_learner import COMALearner
from .potential_q_learner import  PotentialQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["potential_q_learner"] = PotentialQLearner
