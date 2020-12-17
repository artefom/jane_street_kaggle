import os
import shutil
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

__all__ = ['run', 'Module']


class Module:

    def transform(self, data: pd.DataFrame):
        raise NotImplementedError()

    @property
    def dask_before(self):
        return 0


def _run_fit_dask(dataset: dd.DataFrame, model_step: Module):
    if 'partial_fit' not in model_step.__class__.__dict__:  # Skip fitting if partial_fit does not exists
        assert 'fit' not in model_step.__class__.__dict__, "%s must contain partial_fit" % model_step.__class__.__name__
        return None

    def fit_wrap(dataset):  # Use wrapper to avoid returning None
        model_step.partial_fit(dataset)
        return pd.DataFrame([[1]], columns=['dummy'])  # Return dummy value

    rv = dataset.map_overlap(func=fit_wrap,
                             before=model_step.dask_before,
                             after=0,
                             meta=(('dummy', np.int),))
    rv.compute()


def _run_fit_pandas(dataset: pd.DataFrame, model_step: Module):
    if 'partial_fit' not in model_step.__class__.__dict__:  # Skip fitting if partial_fit does not exists
        assert 'fit' not in model_step.__class__.__dict__, "%s must contain partial_fit" % model_step.__class__.__name__
        return None
    model_step.partial_fit(dataset)


def _run_transform_dask(dataset: dd.DataFrame, model_step: Module) -> dd.DataFrame:
    rv = dataset.map_overlap(func=model_step.transform,
                             before=model_step.dask_before,
                             after=0)
    return rv


def _run_transform_pandas(dataset: pd.DataFrame, model_step: Module) -> pd.DataFrame:
    return model_step.transform(dataset)


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


def _run_pandas(dataset: pd.DataFrame, pipeline: list, fit: bool) -> pd.DataFrame:
    dataset = dataset.copy()
    for stage_i, (stage_name, func) in enumerate(pipeline):
        print("Stage, Pandas [{}/{}] - {}".format(stage_i + 1, len(pipeline), stage_name))
        if fit:  # Fit only when step needs to be fitted and current run is fit run
            _run_fit_pandas(dataset, func)
        dataset = _run_transform_pandas(dataset, func)
    return dataset


def _run_dask(dataset: dd.DataFrame, pipeline: list, fit: bool, cache_dir: str) -> dd.DataFrame:
    for stage_i, (stage_name, func) in enumerate(pipeline):
        print("Stage, Dask [{}/{}] - {}".format(stage_i + 1, len(pipeline), stage_name))
        if fit:  # Fit only when step needs to be fitted and current run is fit run
            _run_fit_dask(dataset, func)
        dataset = _run_transform_dask(dataset, func)
        dataset = _persist_to_file(dataset, stage_i, stage_name, cache_dir)
    return dataset


def run(dataset: Union[pd.DataFrame, dd.DataFrame], pipeline: list, fit: bool, cache_dir=None) \
        -> Union[pd.DataFrame, dd.DataFrame]:
    if isinstance(dataset, pd.DataFrame):
        return _run_pandas(dataset, pipeline, fit)
    elif isinstance(dataset, dd.DataFrame):
        return _run_dask(dataset, pipeline, fit, cache_dir)
    else:
        raise NotImplementedError(dataset.__class__.__name__)
