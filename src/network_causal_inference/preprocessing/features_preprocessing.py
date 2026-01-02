import logging

from pandas import DataFrame
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

logger = logging.getLogger(__name__)

def keep_selected_features_cols(df, selected_feature_set: list) -> DataFrame:
    return df[selected_feature_set].copy()

def drop_features_cols(df: DataFrame, drop_col_list: list):
    df.drop(drop_col_list,axis=1,inplace=True)

def drop_duplicates(df: DataFrame):
    logger.info('Total before dropping duplicates: %d',df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    logger.info('Total after dropping duplicates: %d', df.duplicated().sum())

def find_missing_values(df: DataFrame):
    missing_data = df.isnull().sum().to_frame().rename(columns={0: "Total No. of Missing Values"})
    missing_data['% of Missing Values'] = round((missing_data['Total No. of Missing Values'] / len(df)) * 100, 2)
    logger.info('Missing data: %s', missing_data)

def discretize_features(df: DataFrame, continuous_feature_set: list, kn_bins=5) -> DataFrame:
    logger.info('Discretize data for continuous feature set: %s', continuous_feature_set)
    continuous_features_cols = [feature for feature in continuous_feature_set if feature in df.columns]
    logger.info(continuous_features_cols)
    # for continuous_feature in continuous_feature_set:
    if continuous_features_cols:
        logger.info('Discretize data for continuous feature: %s', continuous_features_cols)
        discretizer = KBinsDiscretizer(encode='ordinal', strategy='quantile')
        df[continuous_features_cols] = discretizer.fit_transform(df[continuous_features_cols])
    return df
    # return df.astype(int)

def encode_categorical_features(df: DataFrame, categorical_feature_set: list) -> dict:
    logger.info('Label encoding for categorical feature set: %s', categorical_feature_set)
    label_encoders = dict()
    for categorical_feature in categorical_feature_set:
        if categorical_feature in df.columns:
            le = LabelEncoder()
            df[categorical_feature] = le.fit_transform(df[categorical_feature])
            label_encoders[categorical_feature] = le

    return label_encoders
