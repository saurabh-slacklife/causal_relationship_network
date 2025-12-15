from unittest import TestCase
from features import get_features, drop_features_cols
import logging
logging.basicConfig(level=logging.INFO)

class Test(TestCase):
    def test_load_data(self):
        df = get_features()
        drop_features_cols(df,['id'])
        logging.info('Features: %s', df)