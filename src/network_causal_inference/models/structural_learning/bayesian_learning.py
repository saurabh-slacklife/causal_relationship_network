from pandas import DataFrame
from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass, PriorType
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ParameterEstimator, BayesianEstimator
from pgmpy.inference import Inference, VariableElimination
import networkx as nx
import pandas as pd
import warnings
from pgmpy.global_vars import config
import logging

config.set_backend("numpy")
logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def learn_bayesian_network(df, algorithm=BayesianAlgorithm,
                           score_estimator=ScoringEstimatorClass,
                           max_indegree=None) -> DiscreteBayesianNetwork:
    logger.info('Using Algorithm=%s and ScoreEstimator=%s', algorithm.value, score_estimator.value)

    bayesian_algorithm = algorithm.value(df)

    # bayesian_network_model = DiscreteBayesianNetwork()
    # features = [col for col in df.columns]
    # edges = [(feature, 'label') for feature in features]
    # bayesian_network_model.add_edges_from(edges)

    estimated_bayesian_network_model = bayesian_algorithm.estimate(scoring_method=score_estimator.value(df),
                                                                   start_dag=None,
                                                                   max_indegree=max_indegree,
                                                                   show_progress=True)

    learned_model_edges_len = len(estimated_bayesian_network_model.edges())
    learned_model_nodes_len = len(estimated_bayesian_network_model.nodes())
    logger.info('Learned BayesianNetwork structure with edges=%d, nodes=%d', learned_model_edges_len,
                 learned_model_nodes_len)

    return estimated_bayesian_network_model


def learn_bn_cpds(model: DiscreteBayesianNetwork, df: DataFrame,
                  estimator: ParameterEstimator = BayesianEstimator, prior_type: PriorType = PriorType.bdeu):
    fitted_model=model.fit(
        df,
        estimator=estimator,
        prior_type=prior_type.value,
        # equivalent_sample_size=10
    )
    print(fitted_model.get_cpds())
    return fitted_model


def infer(model, query_vars: list,
          inference_model: Inference = VariableElimination, evidence: dict = None,
          show_progress: bool = True):
    engine = inference_model(model)
    return engine.query(variables=query_vars, show_progress=show_progress, evidence=evidence, )


def compute_structural_importance(model) -> dict:
    di_graph = nx.DiGraph(model.edges())
    result_dict= dict(degree_centrality=nx.degree_centrality(di_graph),
                      betweenness_centrality=nx.betweenness_centrality(di_graph),
                      closeness_centrality=nx.closeness_centrality(di_graph)
                      )
    logger.info('Structurl importance: %s',result_dict)
    return result_dict
