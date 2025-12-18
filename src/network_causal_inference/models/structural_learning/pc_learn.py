from pandas import DataFrame
from pgmpy.estimators import PC

def pc_structural_learn(df: DataFrame, max_indegree=None):
    pc = PC(data=df,
            start_dag=None,
            max_indegree=max_indegree,
            show_progress=True)
    pc_model = pc.estimate()
    return pc_model
