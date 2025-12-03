from unittest import TestCase
from data_loader import load_data
import logging
logging.basicConfig(level=logging.INFO)
class Test(TestCase):
    def test_load_data(self):
        load_data()
