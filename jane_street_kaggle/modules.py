import re

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import *

FEATURE_PATTERN = re.compile(r'^feature_\d+$')
TARGET_PATTERN = re.compile(r'^resp(:?_\d+)?$')


def _get_columns_by_pattern(columns: list, pattern: re.Pattern):
    rv = list()
    for column_name in columns:
        match = pattern.match(column_name)
        if match is None:
            continue
        rv.append(column_name)
    return rv


class Impute(metaclass=PipelineModule):
    """Fill missing values"""

    def __init__(self, impute_value=0):
        self.impute_value = impute_value

    def transform(self, dataset: pd.DataFrame):
        return dataset.fillna(self.impute_value)


class Scale(RandomBatchFit, ParallelTransform, metaclass=PipelineModule):
    """Scale features for better gradient descent"""

    def __init__(self, feature_pattern=FEATURE_PATTERN):
        RandomBatchFit.__init__(self,
                                batch_size=100,
                                seq_size=1,
                                n_epochs=0.1)
        self.feature_pattern = feature_pattern
        self.scaler = None
        self.features = None

    def partial_fit(self, dataset: pd.DataFrame):
        if self.scaler is None:  # First run
            self.features = _get_columns_by_pattern(dataset.columns, self.feature_pattern)
            self.scaler = StandardScaler()
        self.scaler.partial_fit(dataset[self.features])

    def reset_fit(self):
        self.scaler = None
        self.features = None

    def partial_transform(self, dataset: pd.DataFrame):
        dataset[self.features] = self.scaler.transform(dataset[self.features])
        return dataset

    def reset_transform(self):
        pass


class Model(SequentialTransform, metaclass=PipelineModule):
    """Make predictions"""

    def __init__(self, feature_pattern=FEATURE_PATTERN, target_pattern=TARGET_PATTERN):
        self.feature_pattern = feature_pattern
        self.target_pattern = target_pattern
        self.model = None
        self.features = None
        self.target = None

    def partial_transform(self, dataset: pd.DataFrame):
        if self.model is None:
            self.features = _get_columns_by_pattern(dataset.columns, self.feature_pattern)
            self.target = _get_columns_by_pattern(dataset.columns, self.target_pattern)
            self.predicted = ['pred_' + i for i in self.target]
        for pred_col in self.predicted:
            dataset[pred_col] = 0
        return dataset

    def reset_transform(self):
        pass


class Action(SequentialTransform, metaclass=PipelineModule):
    """Get action for given prediction"""

    def __init__(self, threshold=0, target_pattern=TARGET_PATTERN):
        self.target_pattern = target_pattern
        self.threshold = threshold
        self.responses = None
        self.actions = None

    def partial_transform(self, dataset: pd.DataFrame):
        if self.responses is None:
            self.responses = _get_columns_by_pattern(dataset.columns, self.target_pattern)
            self.actions = ['act_' + i for i in self.responses]
        dataset[self.actions] = (dataset[self.responses] > self.threshold).astype(int)
        return dataset

    def reset_transform(self):
        pass
