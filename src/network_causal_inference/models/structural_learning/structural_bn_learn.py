from pandas import DataFrame
from pgmpy.estimators import HillClimbSearch, BIC,PC
from pgmpy.base import DAG
import logging

logger = logging.getLogger(__name__)

def hill_climb_structural_learn(df: DataFrame, max_indegree=None) -> DAG:
    hc = HillClimbSearch(df)
    hc_model = hc.estimate(scoring_method=BIC(df),
                        start_dag=None,
                        max_indegree=max_indegree,
                        show_progress=True)
    learned_model_edges_len = len(hc_model.edges())
    learned_model_nodes_len = len(hc_model.nodes())
    logger.info('Learned BayesianNetwork_HC_Algo structure with edges=%d, nodes=%d', learned_model_edges_len,
                learned_model_nodes_len)

    return hc_model

def pc_structural_learn(df: DataFrame, max_indegree=None) -> DAG:
    pc = PC(data=df,
            start_dag=None,
            max_indegree=max_indegree,
            show_progress=True)
    pc_model = pc.estimate()

    learned_model_edges_len = len(pc_model.edges())
    learned_model_nodes_len = len(pc_model.nodes())
    logger.info('Learned BayesianNetwork_PC_Algo structure with edges=%d, nodes=%d', learned_model_edges_len,
                learned_model_nodes_len)

    return pc_model

