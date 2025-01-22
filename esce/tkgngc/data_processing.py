import torch
import pandas as pd

class TKGNGCDataProcessor:
    def __init__(self, data, device, num_timestamps=20, lags=1):
        """
        Prepares the data for time-series and temporal knowledge graph models.
        Args:
            data (pd.DataFrame): Input time-series dataframe with a time column and features.
            device (torch.device): Device for tensor computation (e.g., 'cpu' or 'cuda').
            num_timestamps (int): Number of discrete timestamps to create.
            lags (int): Number of lagged relationships to consider.
        """
        self.device = device
        self.data = data
        self.num_timestamps = num_timestamps
        self.lags = lags

        # Ensure the dataframe has a datetime index or time column
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data['time'] = self.data.index
        elif 'time' not in self.data.columns:
            raise ValueError("The dataframe must have a datetime index or a 'time' column.")

        self.feature_columns = self.data.columns.difference(['time'])

        # Create time-series tensor
        self.time_series_tensor = torch.tensor(
            self.data[self.feature_columns].values, dtype=torch.float32, device=self.device
        )

        # Generate timestamps
        self.timestamp_indices = self.generate_timestamps()

        # Entity and relation indices
        self.entity_indices = torch.arange(len(self.time_series_tensor), dtype=torch.long, device=self.device)
        self.relation_indices = self.generate_relation_indices()

        # Edge index with lagged relationships
        self.edge_index = self.generate_edge_index()

    def generate_timestamps(self):
        """
        Create discrete timestamp indices using binning.
        """
        time_column = pd.to_datetime(self.data['time'])
        time_min, time_max = time_column.min(), time_column.max()
        bins = pd.date_range(start=time_min, end=time_max, periods=self.num_timestamps + 1)
        return torch.tensor(
            pd.cut(time_column, bins=bins, labels=False, include_lowest=True), dtype=torch.long, device=self.device
        )

    def generate_relation_indices(self):
        """
        Generate relation indices for edges, matching the number of edges created with lags.
        """
        relations = []
        for i in range(len(self.entity_indices)):
            for lag in range(1, self.lags + 1):
                if i - lag >= 0:
                    # Example: Alternate between two relations for simplicity
                    relations.append(0 if lag % 2 == 0 else 1)
        return torch.tensor(relations, dtype=torch.long, device=self.device)

    def generate_edge_index(self):
        """
        Generate edge indices with lagged relationships.
        """
        edges = []
        for i in range(len(self.entity_indices)):
            for lag in range(1, self.lags + 1):
                if i - lag >= 0:
                    edges.append([i, i - lag])
        return torch.tensor(edges, dtype=torch.long, device=self.device).t()  # Transpose for PyG format
