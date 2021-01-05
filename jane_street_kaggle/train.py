import logging
import os

logger = logging.getLogger(__name__)


def train(fit_data_path: str,
          cache_dir: str = None):
    # Get configuration
    from jane_street_kaggle.configuration import conf
    logger.info("Training model...")
    logger.info("Mlflow url: {}".format(conf.get('tracking', 'mlflow_uri')))

    if cache_dir is None:
        cache_dir = conf.get('cache', 'dir')

    logger.info("Using cache directory: {}".format(cache_dir))

    from jane_street_kaggle import modules
    from jane_street_kaggle.pipeline import run
    import dask.dataframe as dd

    # Define pipeline
    pipeline = [
        ('index', modules.Index()),
        ('scale', modules.Scale()),
        ('impute', modules.Impute(0)),
        ('model', modules.Model()),
    ]

    # Read data
    data_format = os.path.splitext(fit_data_path)[1].lower()
    if data_format == '.csv':
        dat = dd.read_csv(fit_data_path)
    elif data_format == '.parquet':
        dat = dd.read_parquet(fit_data_path)
    else:
        raise NotImplementedError("Unknown data format: %s" % fit_data_path)

    logger.info("Running fit pipeline")

    # Run pipeline
    run(dat, pipeline, True, cache_dir=cache_dir)

    logger.info("Done")
