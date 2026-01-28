from collections import defaultdict
from tabulate import tabulate

from pandas import DataFrame
import unittest
from unittest import TestCase
from data_loader import load_data, describe_data
from network_causal_inference.preprocessing.features_preprocessing import drop_features_cols,drop_duplicates,find_missing_values, keep_selected_features_cols
from network_causal_inference.preprocessing.features_preprocessing import discretize_features,encode_categorical_features
from network_causal_inference.models.structural_learning import bayesian_learning as bn_learning
from network_causal_inference.models.structural_learning import pc_learn
from network_causal_inference.visualization.draw_graph import visualize_network, visualize_network_daft
from network_causal_inference.config.common_enums import BayesianAlgorithm, ScoringEstimatorClass
from network_causal_inference.models.structural_learning import hill_climb_learn as bn_learn
from network_causal_inference.models.structural_learning import compute_metrics
from pgmpy.inference import CausalInference, VariableElimination, DBNInference
import logging
from datetime import datetime
import pprint


log_filename = datetime.now().strftime('causal_inference_log_%H-%M-%S-%d-%m-%Y.log')

logging.basicConfig(
    filename='./../../../data/result/logs/'+log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    force=True
)

logger = logging.getLogger(__name__)

required_feature_set = ['dur', 'rate', 'proto', 'service', 'state', 'spkts', 'dpkts',
                                'sbytes', 'dbytes', 'sttl', 'dttl']

def inference_queries(cpd_learnt_bn_model, q_variables, q_evidence):
    logger.info(
        '********** Inferrene starting for network congestion/degradation for p(%s | %s) ***********', q_variables, q_evidence)
    nw_congestion_cpd = bn_learning.network_congestion_learning(cpd_learnt_bn_model, variables=q_variables,
                                                                evidence=q_evidence)
    logger.info('********** Inferrene for network congestion/degradation for p(%s | %s) is %s ***********', q_variables, q_evidence,
                nw_congestion_cpd)

def feature_processing() -> DataFrame:
        logger.info('********** Load data ***********')
        # feature_set_df = load_data(path='../../../data/features/',encoding='cp1252')
        df = load_data()
        # df.columns = feature_set_df['Name']
        logger.info('********** Describe data ***********')
        describe_data(df)

        logger.info('********** Drop specific columns***********')
        df = keep_selected_features_cols(df, required_feature_set)
        # drop_feature_list = ["id", "attack_cat",'label']
        # drop_features_cols(df,drop_col_list=drop_feature_list)
        logger.info('********** Describe data after dropping ***********')
        describe_data(df)
        logger.info('********** Drop duplicates ***********')
        drop_duplicates(df)
        logger.info('********** Describe data after dropping duplicates ***********')
        logger.info('********** Find missing data that is null/none ***********')
        find_missing_values(df)

        continuous_feature_set = ['dur', 'sbytes', 'dbytes', 'sttl', 'rate', 'dttl']
        categorical_feature_set = ['proto', 'service', 'state']
        df = discretize_features(df, continuous_feature_set)

        label_encoders = encode_categorical_features(df, categorical_feature_set)

        logger.info('********** Label encoders *********** %s', label_encoders)
        describe_data(df)

        return df

class Test(TestCase):

    # def test_load_data(self):
    #     logger.info('********** Load data ***********')
    #     # feature_set_df = load_data(path='../../../data/features/',encoding='cp1252')
    #     df = load_data()
    #     # df.columns = feature_set_df['Name']
    #     logger.info('********** Describe data ***********')
    #     describe_data(df)

    @classmethod
    def setUpClass(cls):
        cls.initial_df = feature_processing()
        cls.comparison_dict = {}

    @classmethod
    def tearDownClass(cls):
        table = tabulate(cls.comparison_dict.items(), tablefmt="grid")
        logger.info("\n%s", table)

    def test_learn_bn_ges(self):
        df = self.initial_df.copy()
        # df = feature_processing().copy()
        logger.info('********** Structural learning: Learn Bayesian network: GES ***********')
        ges_metrics = {}
        dbn_ges_model = bn_learn.ges_global_learn(df)
        ges_metrics['edges']=len(dbn_ges_model.edges)
        ges_metrics['nodes']=len(dbn_ges_model.nodes)

        cpd_dbn_ges_model = bn_learning.learn_bn_cpds(dbn_dag_model=dbn_ges_model,df=df)
        logger.info('********** Learnt CPD BN Model***********\n %s',pprint.pformat(cpd_dbn_ges_model.get_cpds()))

        visualize_network(cpd_dbn_ges_model, save_path='../../../data/result/network_ges_28.png',name='Greedy Equivalence Search')

        bn_learning.save_model(dbn_model=cpd_dbn_ges_model, file_name='dbn_hc_model_ges_28')

        bn_learning.compute_structural_importance(cpd_dbn_ges_model)

        min_i_map = bn_learning.get_feature_i_map(dbn_model=cpd_dbn_ges_model,features=required_feature_set)
        logger.info('********** Minimal I-map ********** \n %s', pprint.pformat(min_i_map))
        v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=cpd_dbn_ges_model)
        logger.info('********** V-structural immoralities **********\n %s',pprint.pformat(v_structural_immoralities))

        logger.info('**************Metrics and Score evalkuation start*****************')

        fisher_score = compute_metrics.compute_fisher_score(model=cpd_dbn_ges_model, data=df)
        logger.info('Fisher_Score:%s', fisher_score)
        ges_metrics['fisher_score'] = fisher_score

        structure_score = compute_metrics.compute_structure_score(model=cpd_dbn_ges_model, data=df)
        logger.info('structure_Score:%s', structure_score)
        ges_metrics['structure_score'] = structure_score

        correlation_score = compute_metrics.compute_correlation_score(model=cpd_dbn_ges_model, data=df)
        logger.info('correlation_score:%s', correlation_score)
        ges_metrics['correlation_score'] = correlation_score

        conditional_independencies_df = compute_metrics.compute_conditional_independencies(model=cpd_dbn_ges_model, data=df)
        logger.info('conditional_independencies:%s', conditional_independencies_df)

        log_likelihood_score = compute_metrics.compute_log_likelihood(model=cpd_dbn_ges_model,
                                                                      data=df)
        logger.info('log_likelihood_score:%s', log_likelihood_score)
        ges_metrics['log_likelihood_score'] = log_likelihood_score

        logger.info('**************Metrics and Score evalkuation end*****************')

        logger.info('**************Parameter Inference*****************')

        q_variables = ['rate']
        q_evidence = {'spkts': 486}
        inference_queries(cpd_dbn_ges_model, q_variables,q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4}
        inference_queries(cpd_dbn_ges_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3}
        inference_queries(cpd_dbn_ges_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3, 'dur': 1}
        inference_queries(cpd_dbn_ges_model, q_variables, q_evidence)
        self.comparison_dict['GES'] = ges_metrics

        # for feature in required_feature_set:
        #     logger.info('********** Inferrene starting for %s ***********', feature)
        #     cpds = bn_learning.infer(
        #         dbn_model=cpd_learnt_bn_model,
        #         inference_model=CausalInference,
        #         query_vars=[feature],
        #         # evidence={"dbytes": 1.0, "sttl": 1.0}
        #     )
        #     logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)

    @unittest.skip("Can't run in parallel, execute serially")
    def test_learn_bn_mm_hill_climb(self):
        # df = feature_processing()
        df = self.initial_df.copy()
        logger.info('********** Structural learning: Learn Bayesian network: Min-Max HillClimb ***********')

        dbn_mm_hc_model = bn_learn.min_max_hill_climb_structural_global_learn(df)

        cpd_dbn_mm_hc_model = bn_learning.learn_bn_cpds(dbn_dag_model=dbn_mm_hc_model,df=df)
        logger.info('********** Learnt CPD BN Model***********\n %s',pprint.pformat(cpd_dbn_mm_hc_model.get_cpds()))

        visualize_network(cpd_dbn_mm_hc_model, save_path='../../../data/result/network_mm_hill_climb_28.png',name='MinMaxHillClimbSearch')

        bn_learning.save_model(dbn_model=cpd_dbn_mm_hc_model, file_name='dbn_hc_model_mm_hill_climb_28')

        bn_learning.compute_structural_importance(cpd_dbn_mm_hc_model)

        min_i_map = bn_learning.get_feature_i_map(dbn_model=cpd_dbn_mm_hc_model,features=required_feature_set)
        logger.info('********** Minimal I-map ********** \n %s', pprint.pformat(min_i_map))
        v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=cpd_dbn_mm_hc_model)
        logger.info('********** V-structural immoralities **********\n %s',pprint.pformat(v_structural_immoralities))

        logger.info('**************Parameter Inference*****************')

        q_variables = ['rate']
        q_evidence = {'spkts': 486}
        inference_queries(cpd_dbn_mm_hc_model, q_variables,q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4}
        inference_queries(cpd_dbn_mm_hc_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3}
        inference_queries(cpd_dbn_mm_hc_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3, 'dur': 1}
        inference_queries(cpd_dbn_mm_hc_model, q_variables, q_evidence)

        # for feature in required_feature_set:
        #     logger.info('********** Inferrene starting for %s ***********', feature)
        #     cpds = bn_learning.infer(
        #         dbn_model=cpd_learnt_bn_model,
        #         inference_model=CausalInference,
        #         query_vars=[feature],
        #         # evidence={"dbytes": 1.0, "sttl": 1.0}
        #     )
        #     logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)

    @unittest.skip("Can't run in parallel, execute serially")
    def test_learn_bn_pc(self):
        # df = feature_processing()
        df = self.initial_df.copy()
        logger.info('********** Structural learning: Learn Bayesian network: PC ***********')
        dbn_pc_model = pc_learn.pc_structural_local_learn(df)

        visualize_network(dbn_pc_model, save_path='../../../data/result/network_pc_28.png',name='PC')

        bn_learning.save_model(dbn_model=dbn_pc_model, file_name='dbn_pc_model_pc_28')

        bn_learning.compute_structural_importance(dbn_pc_model)

        min_i_map = bn_learning.get_feature_i_map(dbn_model=dbn_pc_model,features=required_feature_set)
        logger.info('********** Minimal I-map ********** \n %s', pprint.pformat(min_i_map))
        v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=dbn_pc_model)
        logger.info('********** V-structural immoralities **********\n %s',pprint.pformat(v_structural_immoralities))

        logger.info('**************Metrics and Score evalkuation start*****************')

        fisher_score = compute_metrics.compute_fisher_score(model=dbn_pc_model,data=df)
        logger.info('Fisher_Score:%s',fisher_score)

        structure_score = compute_metrics.compute_structure_score(model=dbn_pc_model, data=df)
        logger.info('structure_Score:%s', structure_score)

        correlation_score = compute_metrics.compute_correlation_score(model=dbn_pc_model, data=df)
        logger.info('correlation_score:%s', correlation_score)

        conditional_independencies_df = compute_metrics.compute_conditional_independencies(model=dbn_pc_model, data=df)
        logger.info('conditional_independencies:%s', conditional_independencies_df)

        log_likelihood_score = compute_metrics.compute_log_likelihood(model=dbn_pc_model,
                                                                                           data=df)
        logger.info('log_likelihood_score:%s', log_likelihood_score)

        logger.info('**************Metrics and Score evalkuation end*****************')


        logger.info('**************Parameter Inference*****************')

        # q_variables = ['rate']
        # q_evidence = {'spkts': 486}
        # inference_queries(cpd_learnt_bn_model, q_variables,q_evidence)
        #
        # q_variables = ['rate']
        # q_evidence = {'dbytes': 4}
        # inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)
        #
        # q_variables = ['rate']
        # q_evidence = {'dbytes': 4, 'sbytes': 3}
        # inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)
        #
        # q_variables = ['rate']
        # q_evidence = {'dbytes': 4, 'sbytes': 3, 'dur': 1}
        # inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        # for feature in required_feature_set:
        #     logger.info('********** Inferrene starting for %s ***********', feature)
        #     cpds = bn_learning.infer(
        #         dbn_model=cpd_learnt_bn_model,
        #         inference_model=CausalInference,
        #         query_vars=[feature],
        #         # evidence={"dbytes": 1.0, "sttl": 1.0}
        #     )
        #     logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)

    def test_learn_bn_hill_climb_k2(self):
        # df = feature_processing()
        df = self.initial_df.copy()
        logger.info('********** Structural learning: Learn Bayesian network: HillClimb with K2 ***********')

        dbn_hc_model = bn_learn.hill_climb_structural_global_learn(df,scoring_method="k2",max_indegree=2)
        # dbn_pc_model = bn_learn.pc_structural_learn(df.copy())
        # visualize_network(bn_model,'../../../data/result/network.png')

        cpd_learnt_bn_model = bn_learning.learn_bn_cpds(dbn_dag_model=dbn_hc_model,df=df)
        logger.info('********** Learnt CPD BN Model***********\n %s',pprint.pformat(cpd_learnt_bn_model.get_cpds()))

        hc_bn_metrics = {}
        hc_bn_metrics['edges'] = len(cpd_learnt_bn_model.edges)
        hc_bn_metrics['nodes'] = len(cpd_learnt_bn_model.nodes)

        visualize_network(cpd_learnt_bn_model, save_path='../../../data/result/network_hill_climb_k2_28.png',name='HillClimbSearch with K2')

        bn_learning.save_model(dbn_model=cpd_learnt_bn_model, file_name='dbn_hc_model_hill_climb_k2_28')

        bn_learning.compute_structural_importance(cpd_learnt_bn_model)

        min_i_map = bn_learning.get_feature_i_map(dbn_model=cpd_learnt_bn_model,features=required_feature_set)
        logger.info('********** Minimal I-map ********** \n %s', pprint.pformat(min_i_map))
        v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=cpd_learnt_bn_model)
        logger.info('********** V-structural immoralities **********\n %s',pprint.pformat(v_structural_immoralities))

        logger.info('**************Metrics and Score evalkuation start*****************')

        fisher_score = compute_metrics.compute_fisher_score(model=cpd_learnt_bn_model, data=df)
        logger.info('Fisher_Score:%s', fisher_score)
        hc_bn_metrics['fisher_score'] = fisher_score

        structure_score = compute_metrics.compute_structure_score(model=cpd_learnt_bn_model, data=df)
        logger.info('structure_Score:%s', structure_score)
        hc_bn_metrics['structure_score'] = structure_score

        correlation_score = compute_metrics.compute_correlation_score(model=cpd_learnt_bn_model, data=df)
        logger.info('correlation_score:%s', correlation_score)
        hc_bn_metrics['correlation_score'] = correlation_score

        conditional_independencies_df = compute_metrics.compute_conditional_independencies(model=cpd_learnt_bn_model,
                                                                                           data=df)
        logger.info('conditional_independencies:%s', conditional_independencies_df)

        log_likelihood_score = compute_metrics.compute_log_likelihood(model=cpd_learnt_bn_model,
                                                                      data=df)
        logger.info('log_likelihood_score:%s', log_likelihood_score)
        hc_bn_metrics['log_likelihood_score'] = log_likelihood_score
        self.comparison_dict['Hill Climbing K2'] = hc_bn_metrics

        logger.info('**************Metrics and Score evalkuation end*****************')


        logger.info('**************Parameter Inference*****************')

        q_variables = ['rate']
        q_evidence = {'spkts': 486}
        inference_queries(cpd_learnt_bn_model, q_variables,q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4}
        inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3}
        inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3, 'dur': 1}
        inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        # for feature in required_feature_set:
        #     logger.info('********** Inferrene starting for %s ***********', feature)
        #     cpds = bn_learning.infer(
        #         dbn_model=cpd_learnt_bn_model,
        #         inference_model=CausalInference,
        #         query_vars=[feature],
        #         # evidence={"dbytes": 1.0, "sttl": 1.0}
        #     )
        #     logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)

    def test_learn_bn_hill_climb(self):
        # df = feature_processing()
        df = self.initial_df.copy()
        logger.info('********** Structural learning: Learn Bayesian network: HillClimb ***********')

        dbn_hc_model = bn_learn.hill_climb_structural_global_learn(df)
        # dbn_pc_model = bn_learn.pc_structural_learn(df.copy())
        # visualize_network(bn_model,'../../../data/result/network.png')

        cpd_learnt_bn_model = bn_learning.learn_bn_cpds(dbn_dag_model=dbn_hc_model,df=df)
        logger.info('********** Learnt CPD BN Model***********\n %s',pprint.pformat(cpd_learnt_bn_model.get_cpds()))

        hc_bn_metrics = {}
        hc_bn_metrics['edges'] = len(cpd_learnt_bn_model.edges)
        hc_bn_metrics['nodes'] = len(cpd_learnt_bn_model.nodes)

        visualize_network(cpd_learnt_bn_model, save_path='../../../data/result/network_hill_climb_28.png',name='HillClimbSearch')

        bn_learning.save_model(dbn_model=cpd_learnt_bn_model, file_name='dbn_hc_model_hill_climb_28')

        bn_learning.compute_structural_importance(cpd_learnt_bn_model)

        min_i_map = bn_learning.get_feature_i_map(dbn_model=cpd_learnt_bn_model,features=required_feature_set)
        logger.info('********** Minimal I-map ********** \n %s', pprint.pformat(min_i_map))
        v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=cpd_learnt_bn_model)
        logger.info('********** V-structural immoralities **********\n %s',pprint.pformat(v_structural_immoralities))

        logger.info('**************Metrics and Score evalkuation start*****************')

        fisher_score = compute_metrics.compute_fisher_score(model=cpd_learnt_bn_model, data=df)
        logger.info('Fisher_Score:%s', fisher_score)
        hc_bn_metrics['fisher_score'] = fisher_score

        structure_score = compute_metrics.compute_structure_score(model=cpd_learnt_bn_model, data=df)
        logger.info('structure_Score:%s', structure_score)
        hc_bn_metrics['structure_score'] = structure_score

        correlation_score = compute_metrics.compute_correlation_score(model=cpd_learnt_bn_model, data=df)
        logger.info('correlation_score:%s', correlation_score)
        hc_bn_metrics['correlation_score'] = correlation_score

        conditional_independencies_df = compute_metrics.compute_conditional_independencies(model=cpd_learnt_bn_model,
                                                                                           data=df)
        logger.info('conditional_independencies:%s', conditional_independencies_df)

        log_likelihood_score = compute_metrics.compute_log_likelihood(model=cpd_learnt_bn_model,
                                                                      data=df)
        logger.info('log_likelihood_score:%s', log_likelihood_score)
        hc_bn_metrics['log_likelihood_score'] = log_likelihood_score
        self.comparison_dict['Hill Climbing'] = hc_bn_metrics

        logger.info('**************Metrics and Score evalkuation end*****************')


        logger.info('**************Parameter Inference*****************')

        q_variables = ['rate']
        q_evidence = {'spkts': 486}
        inference_queries(cpd_learnt_bn_model, q_variables,q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4}
        inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3}
        inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        q_variables = ['rate']
        q_evidence = {'dbytes': 4, 'sbytes': 3, 'dur': 1}
        inference_queries(cpd_learnt_bn_model, q_variables, q_evidence)

        # for feature in required_feature_set:
        #     logger.info('********** Inferrene starting for %s ***********', feature)
        #     cpds = bn_learning.infer(
        #         dbn_model=cpd_learnt_bn_model,
        #         inference_model=CausalInference,
        #         query_vars=[feature],
        #         # evidence={"dbytes": 1.0, "sttl": 1.0}
        #     )
        #     logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)

    # def test_learn_bn(self):
    #     logger.info('********** Load data ***********')
    #     # feature_set_df = load_data(path='../../../data/features/',encoding='cp1252')
    #     df = load_data()
    #     # df.columns = feature_set_df['Name']
    #     logger.info('********** Describe data ***********')
    #     describe_data(df)
    #
    #     required_feature_set = ['dur','rate','proto', 'service', 'state', 'spkts', 'dpkts',
    #                 'sbytes', 'dbytes', 'sttl', 'dttl']
    #
    #     logger.info('********** Drop specific columns***********')
    #     df = keep_selected_features_cols(df,required_feature_set)
    #     # drop_feature_list = ["id", "attack_cat",'label']
    #     # drop_features_cols(df,drop_col_list=drop_feature_list)
    #     logger.info('********** Describe data after dropping ***********')
    #     describe_data(df)
    #     logger.info('********** Drop duplicates ***********')
    #     drop_duplicates(df)
    #     logger.info('********** Describe data after dropping duplicates ***********')
    #     logger.info('********** Find missing data that is null/none ***********')
    #     find_missing_values(df)
    #
    #     continuous_feature_set = ['dur', 'sbytes', 'dbytes', 'sttl','rate','dttl']
    #     categorical_feature_set = ['proto', 'service', 'state']
    #     df = discretize_features(df, continuous_feature_set)
    #
    #     label_encoders=encode_categorical_features(df,categorical_feature_set)
    #
    #     logger.info('********** Label encoders *********** %s',label_encoders)
    #     describe_data(df)
    #     logger.info('********** Structural learning: Learn Bayesian network ***********')
    #
    #     dbn_hc_model = bn_learn.hill_climb_structural_global_learn(df.copy())
    #     # dbn_pc_model = bn_learn.pc_structural_learn(df.copy())
    #     # visualize_network(bn_model,'../../../data/result/network.png')
    #
    #     cpd_learnt_bn_model = bn_learning.learn_bn_cpds(dbn_dag_model=dbn_hc_model,df=df)
    #     logger.info('********** Learnt CPD BN Model: %s ***********',cpd_learnt_bn_model)
    #
    #     visualize_network(cpd_learnt_bn_model, save_path='../../../data/result/network.png',name='HillClimbSearch')
    #
    #     bn_learning.save_model(dbn_model=cpd_learnt_bn_model, file_name='dbn_hc_model')
    #
    #     for feature in required_feature_set:
    #         logger.info('********** Inferrene starting for %s ***********', feature)
    #         cpds = bn_learning.infer(
    #             dbn_model=cpd_learnt_bn_model,
    #             inference_model=CausalInference,
    #             query_vars=[feature],
    #             # evidence={"dbytes": 1.0, "sttl": 1.0}
    #         )
    #         logger.info('********** Inferred CPDs BN Model for %s: %s ***********',feature, cpds)
    #
    #     bn_learning.compute_structural_importance(cpd_learnt_bn_model)
    #
    #
    #     min_i_map = bn_learning.get_feature_i_map(dbn_model=cpd_learnt_bn_model,features=required_feature_set)
    #     logger.info('***minimal I-map *******')
    #     logger.info('%s',min_i_map)
    #
    #     v_structural_immoralities = bn_learning.get_bn_immoralities(dbn_model=cpd_learnt_bn_model)
    #     logger.info('***V-structural immoralities *******')
    #     logger.info('%s', v_structural_immoralities)
    #
    #
    #
    #
    #     logger.info('*** Inferrene starting for network congestion/degradation for P(rate|spkts=high-10646)***********')
    #     q_variables = ['rate']
    #     q_evidence = {'spkts': 486}
    #     nw_congestion_cpd = bn_learning.network_congestion_learning(cpd_learnt_bn_model, variables=q_variables,
    #                                                                 evidence=q_evidence)
    #     logger.info('*** Inferrene starting for network congestion/degradation for P(rate|spkts=high-10646): %s ***********', nw_congestion_cpd)
