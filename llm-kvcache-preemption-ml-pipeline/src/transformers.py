import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelemetrySanitizer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn custom transformer that remediates hardware telemetry artifacts:
    - Replaces negative PCIe counter reset sentinels (-999.0) with NaN.
    - Performs group-wise median imputation partitioned by (cluster_id, model_tier)
      learned strictly on the training split to prevent temporal look-ahead leakage.
    """
    def __init__(self, group_cols=None):
        self.group_cols = group_cols or ['cluster_id', 'model_tier']
        self.target_cols = ['kv_cache_util_pct', 'gpu_compute_util_pct', 'pcie_bw_gbps']

    def fit(self, X, y=None):
        df = X.copy()
        df.loc[df['pcie_bw_gbps'] < 0, 'pcie_bw_gbps'] = np.nan
        self.group_medians_ = df.groupby(self.group_cols)[self.target_cols].median()
        self.global_medians_ = df[self.target_cols].median()
        return self

    def transform(self, X):
        df = X.copy()
        df.loc[df['pcie_bw_gbps'] < 0, 'pcie_bw_gbps'] = np.nan
        df = df.merge(self.group_medians_, on=self.group_cols, how='left', suffixes=('', '_med'))
        for col in self.target_cols:
            df[col] = df[col].fillna(df[f'{col}_med']).fillna(self.global_medians_[col])
            df.drop(columns=[f'{col}_med'], inplace=True)
        return df


class KVCacheFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer for LLM inference workloads:
    - log_token_volume: log1p(prompt_tokens * batch_size)
    - kv_contention_idx: multiplicative interaction between KV residency and token volume
    - mem_compute_ratio: ratio of KV-cache utilization to GPU compute utilization
    - kv_burst_delta: instantaneous KV utilization minus rolling cluster baseline
    - queue_rolling_std: rolling standard deviation of scheduler queue depth
    - decode_prefill_ratio: max_output_tokens / (prompt_tokens + 1)
    """
    def __init__(self, rolling_window=6):
        self.rolling_window = rolling_window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['log_token_volume'] = np.round(np.log1p(df['prompt_tokens'] * df['batch_size']), 4)
        df['kv_contention_idx'] = np.round(
            (df['kv_cache_util_pct'] / 100.0) * (df['log_token_volume'] / 12.0), 4
        )
        df['mem_compute_ratio'] = np.round(
            df['kv_cache_util_pct'] / (df['gpu_compute_util_pct'] + 1.0), 4
        )
        df['kv_rolling_mean'] = df.groupby('cluster_id')['kv_cache_util_pct'].transform(
            lambda s: s.rolling(self.rolling_window, min_periods=1).mean()
        ).round(3)
        df['kv_burst_delta'] = np.round(df['kv_cache_util_pct'] - df['kv_rolling_mean'], 3)
        df['queue_rolling_std'] = df.groupby('cluster_id')['queue_depth'].transform(
            lambda s: s.rolling(self.rolling_window, min_periods=1).std().fillna(0.0)
        ).round(3)
        df['decode_prefill_ratio'] = np.round(
            df['max_output_tokens'] / (df['prompt_tokens'] + 1.0), 4
        )
        return df
