from unittest import TestCase
from data_loader import load_data, describe_data
from network_causal_inference.preprocessing.features_preprocessing import drop_features_cols,drop_duplicates,find_missing_values, keep_selected_features_cols
from network_causal_inference.preprocessing.features_preprocessing import discretize_features,encode_categorical_features
from network_causal_inference.models.structural_learning.bayesian_learning import learn_bayesian_network, learn_bn_cpds, infer, compute_structural_importance
from network_causal_inference.visualization.draw_graph import visualize_network
from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass
from pgmpy.inference import CausalInference, VariableElimination, DBNInference
import logging
from datetime import datetime

log_filename = datetime.now().strftime('causal_inference_log_%H:%M-%d-%m-%Y.log')

logging.basicConfig(
    filename='./../../data/result/logs/'+log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)

class Test(TestCase):
    def test_load_data(self):
        logger.info('********** Load data ***********')
        # feature_set_df = load_data(path='../../../data/features/',encoding='cp1252')
        df = load_data()
        # df.columns = feature_set_df['Name']
        logger.info('********** Describe data ***********')
        describe_data(df)

        required_feature_set = ['dur','rate','proto', 'service', 'state', 'spkts', 'dpkts',
                    'sbytes', 'dbytes', 'sttl', 'dttl']

        logger.info('********** Drop specific columns***********')
        df = keep_selected_features_cols(df,required_feature_set)
        # drop_feature_list = ["id", "attack_cat",'label']
        # drop_features_cols(df,drop_col_list=drop_feature_list)
        logger.info('********** Describe data after dropping ***********')
        describe_data(df)
        logger.info('********** Drop duplicates ***********')
        drop_duplicates(df)
        logger.info('********** Describe data after dropping duplicates ***********')
        logger.info('********** Find missing data that is null/none ***********')
        find_missing_values(df)

        continuous_feature_set = ['dur', 'sbytes', 'dbytes', 'sttl','rate','dttl']
        categorical_feature_set = ['proto', 'service', 'state']
        combined_feature_list = continuous_feature_set + categorical_feature_set
        df = discretize_features(df, continuous_feature_set)

        label_encoders=encode_categorical_features(df,categorical_feature_set)

        logger.info('********** Label encoders *********** %s',label_encoders)
        describe_data(df)
        logger.info('********** Structural learning: Learn Bayesian network ***********')
        bn_model=learn_bayesian_network(df,algorithm=BayesianAlgorithm.hc, score_estimator=ScoringEstimatorClass.bic_d)
        # visualize_network(bn_model,'../../../data/result/network.png')
        cpd_learnt_bn_model = learn_bn_cpds(model=bn_model,df=df)
        logger.info('********** Learnt CPD BN Model: %s ***********',cpd_learnt_bn_model)
        # visualize_network(bn_model, '../../../data/result/network_with_cpd.png')
        for feature in combined_feature_list:
            logger.info('********** Inferrene starting for %s ***********', feature)
            cpds = infer(
                model=cpd_learnt_bn_model,
                inference_model=CausalInference,
                query_vars=[feature],
                # evidence={"dbytes": 1.0, "sttl": 1.0}
            )
            logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)
        compute_structural_importance(cpd_learnt_bn_model)



