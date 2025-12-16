from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass
from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import logging

def learn_bayesian_network(df, algorithm=BayesianAlgorithm, score_estimator=ScoringEstimatorClass, max_indegree=None):
    logging.info('Using Algorithm=%s and ScoreEstimator=%s',algorithm.value, score_estimator.value)

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
    logging.info('Learned BayesianNetwork structure with edges=%d, nodes=%d',learned_model_edges_len, learned_model_nodes_len)

    return estimated_bayesian_network_model