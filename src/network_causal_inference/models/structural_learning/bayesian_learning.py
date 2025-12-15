from pgmpy.estimators import HillClimbSearch, BIC, K2
from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import logging

def learn_bayesian_network(df, algorithm='hc', score_estimator='bic', max_indegree=3):
    logging.info('Using Algorithm=%s and ScoreEstimator=%s',algorithm, score_estimator)

    if score_estimator == 'bic':
        scoring_method = BIC(df)
    else:
        scoring_method = K2(df)

    hc = HillClimbSearch(df)

    bayesian_network_model = DiscreteBayesianNetwork()
    features = [col for col in df.columns if col != 'label']
    edges = [(feature, 'label') for feature in features]
    bayesian_network_model.add_edges_from(edges)

    estimated_bayesian_network_model = hc.estimate(scoring_method=scoring_method,
                            start_dag=bayesian_network_model,
                            max_indegree=max_indegree,
                            show_progress=True)

    learned_model_edges_len = len(estimated_bayesian_network_model.edges())
    learned_model_nodes_len = len(estimated_bayesian_network_model.nodes())
    logging.info('Learned BayesianNetwork structure with edges=%d, nodes=%d',learned_model_edges_len, learned_model_nodes_len)

    return estimated_bayesian_network_model