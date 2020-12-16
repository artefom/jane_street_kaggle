from jange_street_kaggle.configuration import conf

import logging
logger = logging.getLogger(__name__)

def train():
    logger.info("Executing worklfow...")

    logger.info("Mlflow url: {}".format(conf.get('tracking','mlflow_uri')))


if __name__ == "__main__":
    train()
