from .q_learner import QLearner
from .q_learner_tm import QLearnerTM

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_tm"] = QLearnerTM