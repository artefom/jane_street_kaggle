import math
import tempfile

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.dataframe.core import _emulate
from dask.diagnostics import ProgressBar

__all__ = ['PipelineModule', 'SequentialFit',
           'SequentialTransform', 'PartialFit',
           'PartialTransform', 'RandomBatchFit', 'ParallelTransform']


class PipelineModule(type):

    def __new__(mcs, name, bases, attrs):
        return super().__new__(mcs, name, bases, attrs)


class PartialFit:
    def partial_fit(self, dataset):
        raise NotImplementedError()

    def reset_fit(self):
        raise NotImplementedError()


class PartialTransform:
    def partial_transform(self, dataset):
        raise NotImplementedError()

    def reset_transform(self):
        raise NotImplementedError()


class SequentialFit(PartialFit):

    def partial_fit(self, dataset):
        raise NotImplementedError()

    def reset_fit(self):
        raise NotImplementedError()

    def fit(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            self.partial_fit(dataset)
        elif isinstance(dataset, dd.DataFrame):
            def fun_wrap(dataset):
                self.partial_fit(dataset)
                return pd.DataFrame([[1]], columns=['dummy'])

            # Specify single-threaded scheduler explicitly to ensure sequential data feed
            dataset.map_partitions(fun_wrap, meta=(('dummy', np.int),)).compute(scheduler='single-threaded')


class SequentialTransform(PartialTransform):

    def transform(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            dataset = self.partial_transform(dataset)
            return dataset
        elif isinstance(dataset, dd.DataFrame):
            temp_dir = tempfile.mkdtemp()
            # Get meta beforehand
            meta = _emulate(self.partial_transform, dataset, udf=True)
            self.reset_transform()  # Reset transform before applying to full dataset
            # Apply to full dataset
            dataset.map_partitions(self.partial_transform, meta=meta) \
                .to_parquet(temp_dir, compute=False, engine='fastparquet') \
                .compute(scheduler='single-threaded')
            return temp_dir


class ParallelTransform(PartialTransform):

    def transform(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            dataset = self.partial_transform(dataset)
            return dataset
        elif isinstance(dataset, dd.DataFrame):
            return dataset.map_partitions(self.partial_transform)


class RandomBatchFit(PartialFit):

    def __init__(self,
                 batch_size,
                 seq_size,
                 ):
        self._batch_fit_batch_size = batch_size
        self._batch_fit_seq_size = seq_size

    def _get_random_index(self, dataset):
        if isinstance(dataset, dd.DataFrame):
            index = dataset.index.drop_duplicates().compute().values
        else:
            index = dataset.index.drop_duplicates().values
        index_orig = index.copy()
        index_new = index.copy()
        np.random.shuffle(index_new[:(len(index) // self._batch_fit_seq_size)
                                     * self._batch_fit_seq_size].reshape((-1, self._batch_fit_seq_size)))
        return pd.Series(index_new, index=index_orig)

    def fit(self, dataset):

        dataset = dataset.copy()

        new_index = self._get_random_index(dataset)
        n_batches = int(math.ceil(len(new_index) / self._batch_fit_batch_size))
        orig_index_name = dataset.index.name
        new_index_name = str(orig_index_name) + '_new'

        assert orig_index_name not in dataset.columns
        assert new_index_name not in dataset.columns

        dataset[new_index_name] = new_index
        dataset = dataset.reset_index().set_index(new_index_name)

        if isinstance(dataset, dd.DataFrame):
            dataset = dataset.repartition(npartitions=n_batches)

        with tempfile.TemporaryDirectory() as temp_dir:
            if isinstance(dataset, dd.DataFrame):
                with ProgressBar():
                    dataset.to_parquet(temp_dir, engine='fastparquet')
                dataset = dd.read_parquet(temp_dir)

            if isinstance(dataset, dd.DataFrame):
                def fun_wrap(dataset):
                    self.partial_fit(dataset)
                    return pd.DataFrame([[1]], columns=['dummy'])

                # Specify single-threaded scheduler explicitly to ensure sequential data feed
                dataset.map_partitions(fun_wrap, meta=(('dummy', np.int),)).compute(scheduler='single-threaded')
            else:
                self.partial_fit(dataset)
