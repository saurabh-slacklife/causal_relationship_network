from enum import Enum
from pgmpy.estimators import HillClimbSearch, GES, PC, TreeSearch, ExpertInLoop, MmhcEstimator
from pgmpy.estimators import BIC, K2


class BayesianAlgorithm(Enum):
    pc = PC
    hc = HillClimbSearch
    ges = GES
    tree_search = TreeSearch
    expert_in_loop = ExpertInLoop
    mmhc_estimator = MmhcEstimator


class ScoringEstimatorClass(Enum):
    bic_d = BIC
    k2 = K2


class PriorType(Enum):
    dirichlet = 'dirichlet'
    bdeu = 'BDeu'
    k2 = 'K2'
