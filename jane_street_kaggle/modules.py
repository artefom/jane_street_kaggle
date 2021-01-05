import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import PipelineModule, ParallelTransform, RandomBatchFit, SequentialTransform
from .column_selection import get_columns_by_pattern, FEATURE_PATTERN, TARGET_PATTERN
from .model import Model  # Load model from separate file for better readability

# List of all available modules defined here
__all__ = [
    'Impute',
    'Scale',
    'Model',
    'Action'
]


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
                                batch_size=10_000,
                                seq_size=1,
                                n_epochs=1)
        self.feature_pattern = feature_pattern
        self.scaler = None
        self.features = None

    def partial_fit(self, dataset: pd.DataFrame):
        if self.scaler is None:  # First run
            self.features = get_columns_by_pattern(dataset.columns, self.feature_pattern)
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


class Action(SequentialTransform, metaclass=PipelineModule):
    """Get action for given prediction"""

    def __init__(self, threshold=0, target_pattern=TARGET_PATTERN):
        self.target_pattern = target_pattern
        self.threshold = threshold
        self.responses = None
        self.actions = None

    def partial_transform(self, dataset: pd.DataFrame):
        if self.responses is None:
            self.responses = get_columns_by_pattern(dataset.columns, self.target_pattern)
            self.actions = ['act_' + i for i in self.responses]
        dataset[self.actions] = (dataset[self.responses] > self.threshold).astype(int)
        return dataset

    def reset_transform(self):
        pass


class Index(metaclass=PipelineModule):
    """Index data by timestamp"""

    def __init__(self, index_col='ts_id', partition_size='300M'):
        self.index_col = index_col
        self.partition_size = partition_size

    def transform(self, dataset: pd.DataFrame):
        if dataset.index.name == self.index_col:
            return dataset
        return dataset.set_index(self.index_col).repartition(partition_size=self.partition_size)
