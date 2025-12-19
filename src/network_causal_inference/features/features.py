import logging
from network_causal_inference.preprocessing.data_loader import describe_data, load_data

logger = logging.getLogger(__name__)

def get_features():
    df = load_data(path='../../../data/features/NUSW-NB15_features.csv',encoding='cp1252')
    logger.info('Features: %s',df)
    return df

def drop_features_cols(df, drop_col_list: list):
    df.drop(drop_col_list,axis=1,inplace=True)