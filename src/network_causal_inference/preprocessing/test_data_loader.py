from unittest import TestCase
from data_loader import load_data, describe_data
from network_causal_inference.preprocessing.features_preprocessing import drop_features_cols,drop_duplicates,find_missing_values, keep_selected_features_cols
from network_causal_inference.preprocessing.features_preprocessing import discretize_features,encode_categorical_features
from network_causal_inference.models.structural_learning.bayesian_learning import learn_bayesian_network
from network_causal_inference.visualization.draw_graph import visualize_network
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
                    'sbytes', 'dbytes', 'sttl', 'dttl', 'label']

        logging.info('********** Drop specific columns***********')
        df = keep_selected_features_cols(df,required_feature_set)
        # drop_features_cols(df,['id'])
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
        bn_model=learn_bayesian_network(df)
        visualize_network(bn_model,'../../../data/result/network.png')


