from pandas import DataFrame
from pgmpy.estimators import HillClimbSearch, BIC

def hill_climb_structural_learn(df: DataFrame, max_indegree=None):
    hc = HillClimbSearch(df)
    hc_model = hc.estimate(scoring_method=BIC(df),
                        start_dag=None,
                        max_indegree=max_indegree,
                        show_progress=True)
    return hc_model
