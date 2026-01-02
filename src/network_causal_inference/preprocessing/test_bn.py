from unittest import TestCase
from data_loader import load_data, describe_data
from network_causal_inference.preprocessing.features_preprocessing import drop_features_cols,drop_duplicates,find_missing_values, keep_selected_features_cols
from network_causal_inference.preprocessing.features_preprocessing import discretize_features,encode_categorical_features
from network_causal_inference.models.structural_learning import bayesian_learning as bn_learning
from network_causal_inference.visualization.draw_graph import visualize_network, visualize_network_daft
from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass
from network_causal_inference.models.structural_learning import structural_bn_learn as bn_learn
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

    def test_learn_bn(self):
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
        df = discretize_features(df, continuous_feature_set)

        label_encoders=encode_categorical_features(df,categorical_feature_set)

        logger.info('********** Label encoders *********** %s',label_encoders)
        describe_data(df)
        logger.info('********** Structural learning: Learn Bayesian network ***********')

        dbn_hc_model = bn_learn.hill_climb_structural_learn(df.copy())
        # dbn_pc_model = bn_learn.pc_structural_learn(df.copy())
        # visualize_network(bn_model,'../../../data/result/network.png')

        cpd_learnt_bn_model = bn_learning.learn_bn_cpds(dbn_dag_model=dbn_hc_model,df=df)
        logger.info('********** Learnt CPD BN Model: %s ***********',cpd_learnt_bn_model)

        visualize_network(cpd_learnt_bn_model, save_path='../../../data/result/network.png',name='HillClimbSearch')

        bn_learning.save_model(dbn_model=cpd_learnt_bn_model, file_name='dbn_hc_model')

        for feature in required_feature_set:
            logger.info('********** Inferrene starting for %s ***********', feature)
            cpds = bn_learning.infer(
                dbn_model=cpd_learnt_bn_model,
                inference_model=CausalInference,
                query_vars=[feature],
                # evidence={"dbytes": 1.0, "sttl": 1.0}
            )
            logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)

        bn_learning.compute_structural_importance(cpd_learnt_bn_model)


        min_i_map = bn_learning.get_feature_i_map(dbn_model=cpd_learnt_bn_model,features=required_feature_set)
        logger.info('***minimal I-map *******')
        logger.info('%s',min_i_map)

        v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=cpd_learnt_bn_model)
        logger.info('***V-structural immoralities *******')
        logger.info('%s', v_structural_immoralities)




        logger.info('*** Inferrene starting for network congestion/degradation for P(rate|spkts=high-10646)***********')
        nw_congestion_cpd=bn_learning.network_congestion_learning(cpd_learnt_bn_model)
        logger.info('*** Inferrene starting for network congestion/degradation for P(rate|spkts=high-10646): %s ***********', nw_congestion_cpd)


