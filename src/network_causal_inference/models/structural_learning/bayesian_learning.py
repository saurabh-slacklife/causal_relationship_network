from pandas import DataFrame

from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass, PriorType
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ParameterEstimator, BayesianEstimator
from pgmpy.inference import Inference, VariableElimination
import networkx as nx
import pandas as pd
import warnings
from pgmpy.global_vars import logger, config

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import logging

logger.setLevel(logging.INFO)
config.set_backend("numpy")


def learn_bayesian_network(df, algorithm=BayesianAlgorithm,
                           score_estimator=ScoringEstimatorClass,
                           max_indegree=None) -> DiscreteBayesianNetwork:
    logging.info('Using Algorithm=%s and ScoreEstimator=%s', algorithm.value, score_estimator.value)

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
    logging.info('Learned BayesianNetwork structure with edges=%d, nodes=%d', learned_model_edges_len,
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


def infer(model, query_vars: list, evidence: dict = None):
    engine = VariableElimination(model)
    return engine.query(query_vars, evidence=evidence)


def compute_structural_importance(model):
    di_graph = nx.DiGraph(model.edges())
    logging.debug('Structurl importance: "degree:%f", "betweenness:%f", "closeness:%f",',
                  nx.degree_centrality(di_graph), nx.betweenness_centrality(di_graph),
                  nx.closeness_centrality(di_graph))
    return {
        "degree": nx.degree_centrality(di_graph),
        "betweenness": nx.betweenness_centrality(di_graph),
        "closeness": nx.closeness_centrality(di_graph)
    }
