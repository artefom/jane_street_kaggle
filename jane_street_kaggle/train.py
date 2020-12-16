import logging

from jane_street_kaggle.configuration import conf

logger = logging.getLogger(__name__)


def train(data_path):
    logger.info("Training model...")
    logger.info("Mlflow url: {}".format(conf.get('tracking', 'mlflow_uri')))