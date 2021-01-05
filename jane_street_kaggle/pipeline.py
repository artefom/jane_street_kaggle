import logging
import os
import shutil
from typing import Union

import dask.dataframe as dd
import pandas as pd

__all__ = ['run']

logger = logging.getLogger(__name__)


def _get_cache_path(base_dir, stage_i, stage_name):
    cache_dirname = 'stage_{:0>1}_{}.parquet'.format(stage_i, stage_name)
    cache_path = os.path.join(base_dir, cache_dirname)
    if os.path.exists(cache_path):
        assert os.path.isdir(cache_path), "{} is not a directory. cannot remove".format(cache_path)
        shutil.rmtree(cache_path)
    return cache_path


def _persist_to_file(dataset: Union[str, dd.DataFrame], stage_i, stage_name, cache_dir):
    assert cache_dir is not None, "When using dask dataframe, cache dir must be provided"
    assert '/' not in stage_name \
           and '\\' not in stage_name \
           and ':' not in stage_name \
           and '..' not in stage_name, "Unsafe stage symbols"
    cache_path = _get_cache_path(cache_dir, stage_i, stage_name)
    if isinstance(dataset, dd.DataFrame):
        dataset.to_parquet(cache_path, engine='fastparquet')
    elif isinstance(dataset, str):
        logger.debug("Moving {} to {}".format(dataset, cache_path))
        shutil.move(dataset, cache_path)
    else:
        raise NotImplementedError()

    return dd.read_parquet(cache_path)


def run(dataset: Union[pd.DataFrame, dd.DataFrame], pipeline: list, fit: bool, cache_dir: str) -> dd.DataFrame:
    for step_idx, (name, transformer) in enumerate(pipeline):
        logger.info("Step [{}/{}] - {}".format(step_idx + 1, len(pipeline), name))
        if fit:
            fit_fun = getattr(transformer, 'fit', None)
            if fit_fun is not None:
                reset_fit = getattr(transformer, 'reset_fit', None)
                if reset_fit is not None:  # For partial steps, need to reset before fitting
                    reset_fit()
                fit_fun(dataset)
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.copy()  # Preserve original dataset from inplace modifications.
            dataset_before = None
        else:
            dataset_before = dataset

        # Need to flush internal state of some modules to transform data
        reset_transform = getattr(transformer, 'reset_transform', None)
        if reset_transform is not None:
            reset_transform()

        dataset = transformer.transform(dataset)
        if isinstance(dataset, str) or (isinstance(dataset, dd.DataFrame) and dataset is not dataset_before):
            dataset = _persist_to_file(dataset, step_idx, name, cache_dir)
    return dataset
