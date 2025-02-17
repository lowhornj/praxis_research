import torch
import pandas as pd

class TKGNGCDataProcessor:
    def __init__(self, data, device, num_timestamps=20, lags=1):
        """
        Args:
            data (pd.DataFrame): Time-series data with a 'time' column and feature columns.
            device (torch.device): Computation device ('cpu' or 'cuda').
            num_timestamps (int): Number of discrete time intervals.
            lags (int): Number of lagged relationships to consider.
        """
        self.device = device
        self.data = data
        self.num_timestamps = num_timestamps
        self.lags = lags

        # Check for time column or datetime index
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data['time'] = self.data.index
        elif 'time' not in self.data.columns:
            raise ValueError("Data must have a datetime index or 'time' column.")

        self.feature_columns = self.data.columns.difference(['time'])
        self.num_entities = len(self.feature_columns)

        # Tensor representation of time-series features
        self.time_series_tensor = torch.tensor(
            self.data[self.feature_columns].values, dtype=torch.float32, device=self.device
        )

        # Generate timestamp bins (discretized time)
        self.timestamp_indices = self.generate_timestamps()

        # Generate individual components for quads
        self.head_indices, self.relation_indices, self.tail_indices, self.time_indices = self.generate_quad_components()

    def generate_timestamps(self):
        """
        Converts timestamps to discrete time bins.
        """
        time_column = pd.to_datetime(self.data['time'])
        time_min, time_max = time_column.min(), time_column.max()
        bins = pd.date_range(start=time_min, end=time_max, periods=self.num_timestamps + 1)
        return torch.tensor(
            pd.cut(time_column, bins=bins, labels=False, include_lowest=True),
            dtype=torch.long, device=self.device
        )

    def generate_quad_components(self):
        """
        Generates separate tensors for heads, relations, tails, and timestamps.
        Each quad represents (Var_i, Lag_k, Var_j, Timestamp_t)
        """
        head_list = []
        relation_list = []
        tail_list = []
        time_list = []

        for t in range(self.lags, len(self.time_series_tensor)):  # Start from `lags` to avoid negative indices
            current_time_bin = self.timestamp_indices[t].item()

            for lag in range(1, self.lags + 1):
                past_t = t - lag

                for i, var_i in enumerate(self.feature_columns):  # Head entity (Var_i)
                    for j, var_j in enumerate(self.feature_columns):  # Tail entity (Var_j)

                        # (Var_i at t-lag) → (Var_j at t) with "Lag_k" relation at "time_bin"
                        head_list.append(i)
                        relation_list.append(lag - 1)  # Map lags to indices (0, 1, 2, ...)
                        tail_list.append(j)
                        time_list.append(current_time_bin)

        # Convert lists to tensors
        head_tensor = torch.tensor(head_list, dtype=torch.long, device=self.device)
        relation_tensor = torch.tensor(relation_list, dtype=torch.long, device=self.device)
        tail_tensor = torch.tensor(tail_list, dtype=torch.long, device=self.device)
        time_tensor = torch.tensor(time_list, dtype=torch.long, device=self.device)

        return head_tensor, relation_tensor, tail_tensor, time_tensor

