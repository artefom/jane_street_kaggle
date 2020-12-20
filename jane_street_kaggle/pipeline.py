import os
import shutil
from typing import Union

import dask.dataframe as dd
import pandas as pd

__all__ = ['run']


def _persist_to_file(dataset: dd.DataFrame, stage_i, stage_name, cache_dir):
    assert isinstance(dataset, dd.DataFrame)
    assert cache_dir is not None, "When using dask dataframe, cache dir must be provided"
    assert '/' not in stage_name \
           and '\\' not in stage_name \
           and ':' not in stage_name \
           and '..' not in stage_name, "Unsafe stage symbols"
    cache_filename = 'stage_{:0>1}_{}.parquet'.format(stage_i, stage_name)
    cache_path = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_path):
        assert os.path.isdir(cache_path), "{} is not a directory. cannot remove".format(cache_path)
        shutil.rmtree(cache_path)
    dataset.to_parquet(cache_path, engine='fastparquet', compression='gzip')
    return dd.read_parquet(cache_path)


def run(dataset: Union[pd.DataFrame, dd.DataFrame], pipeline: list, fit: bool, cache_dir: str,
        transform_reset=True) -> dd.DataFrame:
    for step_idx, (name, transformer) in enumerate(pipeline):
        print("Step [{}/{}] - {}".format(step_idx + 1, len(pipeline), name))
        if fit:
            fit_fun = getattr(transformer, 'fit', None)
            if fit_fun is not None:
                reset_fit = getattr(transformer, 'reset_fit', None)
                if reset_fit is not None:  # For partial steps, need to reset before fitting
                    reset_fit()
                fit_fun(dataset)
        dataset_before = dataset if isinstance(dataset, dd.DataFrame) else None
        if transform_reset:  # Reset before transforming data
            reset_transform = getattr(transformer, 'reset_transform', None)
            if reset_transform is not None:
                reset_transform()
        dataset = transformer.transform(dataset)
        if isinstance(dataset, dd.DataFrame) and dataset is not dataset_before:
            dataset = _persist_to_file(dataset, step_idx, name, cache_dir)
    return dataset
