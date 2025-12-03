import pandas as pd
from pandas import DataFrame

import logging
logger = logging.getLogger(__name__)

def load_data(path='../../../data/train/UNSW_NB15_training-set.csv') -> DataFrame:
    df = pd.read_csv(path)
    logger.info('Shaped: %s Features: %s',df.shape, df.columns)
    return df