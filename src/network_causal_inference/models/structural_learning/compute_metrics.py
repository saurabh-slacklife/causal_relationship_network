from pgmpy.metrics import correlation_score, implied_cis, log_likelihood_score, structure_score, fisher_c
from pgmpy.estimators.CITests import chi_square
from pandas import DataFrame

# A higher score represents a better fit
def compute_structure_score(model, data: DataFrame, scoring_method="bic-g") -> float:
    return structure_score(model, data, scoring_method=scoring_method)

# Computes the log-likelihood of a given dataset i.e. P(data | model).
#
# How well the specified model describes the data. High score better fit
def compute_log_likelihood(model, data: DataFrame) -> float:
    return log_likelihood_score(model, data)

def compute_conditional_independencies(model, data: DataFrame, ci_test=chi_square) -> DataFrame:
    return implied_cis(model, data, ci_test=ci_test)

# how well the model structure represents the correlations in the data
# uses this d-connection/d-separation property to compare the model. Based on f1-score
def compute_correlation_score(model, data: DataFrame, test="chi_square", significance_level=0.05) -> float:
    return correlation_score(model, data, test=test, significance_level=significance_level)


def compute_fisher_score(model, data: DataFrame, ci_test=chi_square) -> float:
    return fisher_c(model, data, ci_test=ci_test, )