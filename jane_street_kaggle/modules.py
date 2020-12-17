import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import pipeline


class Impute(pipeline.Module):
    """Fill missing values"""

    def __init__(self, impute_value):
        self.impute_value = impute_value

    def transform(self, dataset: pd.DataFrame):
        return dataset.fillna(self.impute_value)


class Scale(pipeline.Module):
    """Scale features for better gradient descent"""

    def __init__(self):
        self.scaler = None
        self.features = None

    def partial_fit(self, dataset: pd.DataFrame):
        if self.scaler is None:  # First run
            self.features = [i for i in dataset.columns if 'feature' in i]
            self.scaler = StandardScaler()
        self.scaler.partial_fit(dataset[self.features])

    def transform(self, dataset: pd.DataFrame):
        dataset[self.features] = self.scaler.transform(dataset[self.features])
        return dataset


class Model(pipeline.Module):
    """Make predictions"""

    def __init__(self):
        self.model = None
        self.features = None
        self.target = None

    def transform(self, dataset: pd.DataFrame):
        if self.model is None:
            self.features = [i for i in dataset.columns if 'feature' in i]
            self.target = [i for i in dataset.columns if 'resp' in i]
            self.predicted = ['pred_' + i for i in self.target]
        dataset[self.predicted] = np.random.uniform(-1, 1, size=(len(dataset), len(self.predicted)))
        return dataset


class Action(pipeline.Module):
    """Get action for given prediction"""

    def __init__(self, threshold):
        self.threshold = threshold
        self.responses = None
        self.actions = None

    def transform(self, dataset: pd.DataFrame):
        if self.responses is None:
            self.responses = [i for i in dataset.columns if 'resp' in i]
            self.actions = ['act_' + i for i in self.responses]
        dataset[self.actions] = (dataset[self.responses] > self.threshold).astype(int)
        return dataset


class Score(pipeline.Module):
    """Compute score for dataset"""

    def __init__(self):
        pass

    def transform(self, dataset: pd.DataFrame):
        pass
