import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import *
from .column_selection import FEATURE_PATTERN, TARGET_PATTERN, get_columns_by_pattern


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class Model(RandomBatchFit,
            ParallelTransform,
            metaclass=PipelineModule):
    """Make predictions"""

    def __init__(self, feature_pattern=FEATURE_PATTERN, target_pattern=TARGET_PATTERN):
        # Initialize parameters for random batch fit
        RandomBatchFit.__init__(
            self,
            batch_size=10_00,
            seq_size=1,
            n_epochs=100,
        )
        self.feature_pattern = feature_pattern
        self.target_pattern = target_pattern
        self.model = None
        self.feature_names = None
        self.target_names = None

    def partial_transform(self, dataset: pd.DataFrame):
        # Convert features to torch
        features = self._to_torch(dataset[self.feature_names].values)
        # Make predictions
        with torch.no_grad():
            pred = self.model.eval()(features).detach().numpy()
        # Add predictions to dataframe
        dataset[self.prediction_names] = pred
        return dataset

    def reset_transform(self):
        pass

    def _to_torch(self, arr):
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def _need_init_model(self):
        return self.model is None

    def _reset_init_model(self):
        self.model = None

    def _init_model(self, columns):
        self.feature_names = get_columns_by_pattern(columns, self.feature_pattern)  # Column names to get features from
        self.target_names = get_columns_by_pattern(columns, self.target_pattern)  # Column names to get target from
        self.prediction_names = ['pred_' + i for i in self.target_names]  # Column names to store prediction

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model and put it to device
        self.model = NeuralNet(
            input_size=len(self.feature_names),
            hidden_size=len(self.feature_names),
            output_size=len(self.target_names),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001,
                                          weight_decay=0.01)

    def partial_fit(self, dataset: pd.DataFrame):
        assert isinstance(dataset, pd.DataFrame)
        if self._need_init_model():
            self._init_model(dataset.columns)

        features = self._to_torch(dataset[self.feature_names].values)
        target = self._to_torch(dataset[self.target_names].values)

        pred = self.model.train()(features)
        loss = ((target - pred) ** 2).sum()  # Compute loss for gradient calculation

        # TODO: Make proper overfit detection and stop criterion
        # Compute and print score
        with torch.no_grad():
            y_test_pred = self.model.eval()(features).detach().numpy()
            y_test_true = target.detach().numpy()
            score = ((y_test_pred - y_test_true) ** 2).mean()
            print("\rMSE: {:.5f}   ".format(score), end='')

        # Recompute gradient
        self.optimizer.zero_grad()
        loss.backward()

        # Adjust model parameters
        self.optimizer.step()

    def reset_fit(self):
        # Reset everything before next fit attempt
        self._reset_init_model()
