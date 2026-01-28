from pandas import DataFrame
from pgmpy.estimators import HillClimbSearch, BIC,MmhcEstimator, BDeu, GES, K2
from pgmpy.base import DAG
from pgmpy.estimators import MaximumLikelihoodEstimator
import logging

logger = logging.getLogger(__name__)

def hill_climb_structural_global_learn(df: DataFrame, scoring_method="bic-d", max_indegree=None) -> DAG:
    hc = HillClimbSearch(df)
    hc_model = hc.estimate(scoring_method=scoring_method,
                        start_dag=None,
                        max_indegree=max_indegree,
                        show_progress=True)
    learned_model_edges_len = len(hc_model.edges())
    learned_model_nodes_len = len(hc_model.nodes())
    logger.info('Learned BayesianNetwork_HC_Algo structure with edges=%d, nodes=%d', learned_model_edges_len,
                learned_model_nodes_len)
    return hc_model

def min_max_hill_climb_structural_global_learn(df: DataFrame, significance_level=0.01) -> DAG:
    mm_hc = MmhcEstimator(df)
    mm_hc_model = mm_hc.estimate()
    learned_model_edges_len = len(mm_hc_model.edges())
    learned_model_nodes_len = len(mm_hc_model.nodes())
    logger.info('Learned BayesianNetwork_MM_HC_Algo structure with edges=%d, nodes=%d', learned_model_edges_len,
                learned_model_nodes_len)
    return mm_hc_model


def ges_global_learn(df: DataFrame) -> DAG:
    ges = GES(df)
    ges_model = ges.estimate(scoring_method="bic-d")
    learned_model_edges_len = len(ges_model.edges())
    learned_model_nodes_len = len(ges_model.nodes())
    logger.info('Learned BayesianNetwork_GES structure with edges=%d, nodes=%d', learned_model_edges_len,
                learned_model_nodes_len)
    return ges_model
