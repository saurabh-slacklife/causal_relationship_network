from pandas import DataFrame
from pgmpy.estimators import PC
from pgmpy.base import DAG
import logging

logger = logging.getLogger(__name__)

def pc_structural_local_learn(df: DataFrame) -> DAG:
    pc = PC(data=df)
    pc_model = pc.estimate(ci_test='chi_square', variant="parallel", max_cond_vars=2,return_type='dag')

    learned_model_edges_len = len(pc_model.edges())
    learned_model_nodes_len = len(pc_model.nodes())
    logger.info('Learned BayesianNetwork_PC_Algo structure with edges=%d, nodes=%d', learned_model_edges_len,
                learned_model_nodes_len)

    return pc_model
