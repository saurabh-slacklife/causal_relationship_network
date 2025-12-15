import glob
import os

import pandas as pd
from pandas import DataFrame
import numpy as np
import logging
logger = logging.getLogger(__name__)

def load_data(path='../../../data/train/',encoding='utf8') -> DataFrame:
    input_csv_files = glob.glob(os.path.join(path, "*.csv"))
    combined_df_list=list()
    for csv_file in input_csv_files:
        df = pd.read_csv(csv_file,encoding=encoding)
        combined_df_list.append(df)
    return pd.concat(combined_df_list)

def describe_data(df: DataFrame):
    if df is not None:
        df_data_types = df.dtypes

        numerical_dtype_count = 0
        categorical_dtype_count = 0

        for column_name, data_type in df_data_types.items():
            if np.issubdtype(data_type, np.number):
                numerical_dtype_count += 1
            else:
                categorical_dtype_count += 1

        logger.info('Shape: %s Features: %s', df.shape, df.columns)
        logger.info('Statistical summary: %s', df.describe(include='all'))
        logger.info('Number of Numerical: %d and categorical: %d features', numerical_dtype_count, categorical_dtype_count)
        logger.info('Features summary: %s', df.info())
        logger.info('Head: %s', df.head())
    else:
        logger.error('Dataframe is None')