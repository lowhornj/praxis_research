import torch
import pandas as pd

class TKGNGCDataProcessor:
    def __init__(self, data, priors, device, num_timestamps=20, lags=1):
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
        self.priors = priors

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
                        if self.priors.iloc[i,j] == 0:
                            continue

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


def reshape_embeddings(head_emb, tail_emb, processor):
    """
    Reshapes the learned embeddings from (quads) into a structured format:
    [num_rows, lags, num_vars, embedding_dim].

    Args:
        head_emb (torch.Tensor): Head embeddings of shape [num_quads, embedding_dim].
        tail_emb (torch.Tensor): Tail embeddings of shape [num_quads, embedding_dim].
        processor (TKGNGCDataProcessor): The data processor instance.

    Returns:
        torch.Tensor: Reshaped tensor of shape [num_rows, lags, num_vars, embedding_dim].
    """

    embedding_dim = head_emb.shape[1]  # Embedding size per entity
    num_rows = len(processor.data)  # Original time series length
    num_vars = processor.num_entities  # Number of variables/entities
    lags = processor.lags  # Number of lags

    # Initialize output tensor [num_rows, lags, num_vars, embedding_dim]
    embeddings_tensor = torch.zeros((num_rows, lags, num_vars, embedding_dim), device=processor.device)

    # Fill in the embeddings using the quads (head, relation, tail, timestamp)
    for idx in range(len(processor.head_indices)):
        head_var = processor.head_indices[idx].item()
        lag = processor.relation_indices[idx].item()  # This is 0, 1, 2... corresponding to lag-1, lag-2, etc.
        tail_var = processor.tail_indices[idx].item()
        timestamp_idx = processor.time_indices[idx].item()

        # Locate corresponding time index in the original sequence
        time_idx = torch.where(processor.timestamp_indices == timestamp_idx)[0]
        if len(time_idx) == 0:
            continue  # Skip if timestamp isn't found

        # Row index is the step at which the quad is relevant (adjusted for lag)
        row_index = time_idx[-1].item()
        if row_index - (lag + 1) < 0:
            continue  # Skip if the index is out of valid range

        # Assign the embedding for the corresponding variable and lag
        embeddings_tensor[row_index, lag, tail_var] = tail_emb[idx]

    return embeddings_tensor
