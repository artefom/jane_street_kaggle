import math
import tempfile
import time
from datetime import datetime
from multiprocessing import Process
from queue import Empty

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.dataframe.core import _emulate

from .macos_queue import MacosQueue

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


def _read_batches(dataset, queue, jobs):
    for seq_i, epoch_n, batch_n, total_batches, index in jobs:
        print("Seq_i = {}".format(seq_i))
        while queue.qsize() >= 3:
            time.sleep(0.01)
        if isinstance(dataset, dd.DataFrame):
            data_batch = dataset.loc[index].compute()
        else:
            data_batch = dataset.loc[index]
        queue.put((seq_i, epoch_n, batch_n, total_batches, data_batch))
    queue.put(None)


class RandomBatchFit(PartialFit):

    def __init__(self,
                 batch_size,
                 seq_size,
                 n_epochs,
                 ):
        self._batch_fit_batch_size = batch_size
        self._batch_fit_seq_size = seq_size
        self._batch_fit_n_epochs = n_epochs

    def _get_index_batches(self, index: np.ndarray):
        assert isinstance(index, np.ndarray)
        true_n_epochs = math.ceil(self._batch_fit_n_epochs)
        for epoch_n in range(true_n_epochs):
            index_seqs = np.random.choice(np.arange(0, self._batch_fit_seq_size)) \
                         + np.arange(0, index.shape[0], step=self._batch_fit_seq_size)
            index_seqs_len = len(index_seqs)
            np.random.shuffle(index_seqs)
            batches_range = range(0, len(index_seqs), self._batch_fit_batch_size)
            batches_range_len = len(batches_range)
            epoch_frac = self._batch_fit_n_epochs - epoch_n
            for batch_idx_i, batch_idx_range in enumerate(batches_range):
                # Handle fractional batches correctly
                yield epoch_n, batch_idx_i, batches_range_len, \
                      np.concatenate(tuple(index[index_seqs[batch_idx]:index_seqs[batch_idx] + self._batch_fit_seq_size]
                                           for batch_idx in range(batch_idx_range,
                                                                  min(index_seqs_len, batch_idx_range
                                                                      + self._batch_fit_batch_size))))
                if batch_idx_i >= int(round(epoch_frac * batches_range_len)) - 1:
                    break

    def _get_batches(self, dataset):
        print("Iterating {} epochs over ".format(self._batch_fit_n_epochs), end='')
        if isinstance(dataset, dd.DataFrame):
            index = dataset.index.drop_duplicates().compute().values
        else:
            index = dataset.index.drop_duplicates().values
        print("{} rows".format(index.shape[0]))

        jobs = list()
        for seq_i, (epoch_n, batch_n, total_batches, index) in enumerate(self._get_index_batches(index)):
            jobs.append((seq_i, epoch_n, batch_n, total_batches, index))

        batch_queue = MacosQueue()
        batch_extractor_process = Process(target=_read_batches, args=(dataset, batch_queue, jobs))

        print("Starting process...")
        batch_extractor_process.start()
        print("OK")
        while True:
            try:
                dat = batch_queue.get(block=True, timeout=0.1)
            except Empty:
                time.sleep(0.01)
                continue

            if dat is None:
                break
            else:
                seq_i, epoch_n, batch_n, total_batches, data_batch = dat
                progress = (epoch_n * total_batches + batch_n + 1) / \
                           max(int(round((self._batch_fit_n_epochs * total_batches))), 1)
                print('\r{:.2f}%   '.format(progress * 100), sep=' ', end='', flush=True)
                yield data_batch
            print()
        batch_extractor_process.join()

        # for epoch_n, batch_n, total_batches, index in self._get_index_batches(index):
        #     progress = (epoch_n * total_batches + batch_n + 1) / \
        #                max(int(round((self._batch_fit_n_epochs * total_batches))), 1)
        #     print('\r{:.2f}%   '.format(progress * 100), sep=' ', end='', flush=True)
        #     if isinstance(dataset, dd.DataFrame):
        #         yield dataset.loc[index].compute()
        #     else:
        #         yield dataset.loc[index]
        # print()

    def fit(self, dataset):
        batch_times = list()
        self.reset_fit()
        batch_start_time = datetime.now()
        skip = True
        for dataset_batch in self._get_batches(dataset):
            batch_end_time = datetime.now()

            assert len(dataset_batch) > 0
            self.partial_fit(dataset_batch)
            fit_end_time = datetime.now()

            get_data_seconds = (batch_end_time - batch_start_time).total_seconds()
            total_round_seconds = (fit_end_time - batch_start_time).total_seconds()
            if not skip:
                batch_times.append(get_data_seconds / total_round_seconds)
            skip = False
            batch_start_time = datetime.now()

        print("Time spent for data loading: {:.2f}%".format(np.mean(batch_times) * 100))
