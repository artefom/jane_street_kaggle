import dask.dataframe as dd
import numpy as np
import pandas as pd


class SequentialFit:

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

            dataset.map_partitions(fun_wrap, meta=(('dummy', np.int),)).compute(scheduler='single-threaded')


class SequentialTransform:

    def partial_transform(self, dataset):
        raise NotImplementedError()

    def reset_transform(self):
        raise NotImplementedError()

    def transform(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            dataset = self.partial_transform(dataset)
        elif isinstance(dataset, dd.DataFrame):
            dataset = dataset.map_overlap(self.partial_transform, 0, 0)
        return dataset


PartialFit = SequentialFit
PartialTransform = SequentialTransform
