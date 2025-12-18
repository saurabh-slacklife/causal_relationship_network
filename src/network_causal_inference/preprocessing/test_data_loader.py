from unittest import TestCase
from data_loader import load_data, describe_data
from network_causal_inference.preprocessing.features_preprocessing import drop_features_cols,drop_duplicates,find_missing_values, keep_selected_features_cols
from network_causal_inference.preprocessing.features_preprocessing import discretize_features,encode_categorical_features
from network_causal_inference.models.structural_learning.bayesian_learning import learn_bayesian_network, learn_bn_cpds, infer, compute_structural_importance
from network_causal_inference.visualization.draw_graph import visualize_network
from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Test(TestCase):
    def test_load_data(self):
        logging.info('********** Load data ***********')
        # feature_set_df = load_data(path='../../../data/features/',encoding='cp1252')
        df = load_data()
        # df.columns = feature_set_df['Name']
        logging.info('********** Describe data ***********')
        describe_data(df)

        required_feature_set = ['dur','rate','proto', 'service', 'state', 'spkts', 'dpkts',
                    'sbytes', 'dbytes', 'sttl', 'dttl']

        logging.info('********** Drop specific columns***********')
        df = keep_selected_features_cols(df,required_feature_set)
        # drop_feature_list = ["id", "attack_cat",'label']
        # drop_features_cols(df,drop_col_list=drop_feature_list)
        logging.info('********** Describe data after dropping ***********')
        describe_data(df)
        logging.info('********** Drop duplicates ***********')
        drop_duplicates(df)
        logging.info('********** Describe data after dropping duplicates ***********')
        logging.info('********** Find missing data that is null/none ***********')
        find_missing_values(df)

        continuous_feature_set = ['dur', 'sbytes', 'dbytes', 'sttl','rate','dttl']
        df = discretize_features(df, continuous_feature_set)

        categorical_feature_set={'proto', 'service', 'state'}
        label_encoders=encode_categorical_features(df,categorical_feature_set)

        logging.info('********** Label encoders *********** %s',label_encoders)
        describe_data(df)
        logging.info('********** Structural learning: Learn Bayesian network ***********')
        bn_model=learn_bayesian_network(df,algorithm=BayesianAlgorithm.hc, score_estimator=ScoringEstimatorClass.bic_d)
        # visualize_network(bn_model,'../../../data/result/network.png')
        cpd_learnt_bn_model = learn_bn_cpds(model=bn_model,df=df)
        print('*************************',cpd_learnt_bn_model)
        visualize_network(bn_model, '../../../data/result/network_with_cpd.png')
        result = infer(
            model=cpd_learnt_bn_model,
            query_vars=["latency"],
            evidence={"packet_rate": 100, "queue_size": 50}
        )
        print('------------------------',result)


