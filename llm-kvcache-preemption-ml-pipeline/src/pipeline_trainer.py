import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix, fbeta_score
)

from transformers import TelemetrySanitizer, KVCacheFeatureEngineer


FEATURE_COLS_NUM = [
    'batch_size', 'prompt_tokens', 'max_output_tokens', 'queue_depth',
    'kv_cache_util_pct', 'gpu_compute_util_pct', 'pcie_bw_gbps',
    'log_token_volume', 'kv_contention_idx', 'mem_compute_ratio',
    'kv_rolling_mean', 'kv_burst_delta', 'queue_rolling_std', 'decode_prefill_ratio'
]
FEATURE_COLS_CAT = ['cluster_id', 'model_tier', 'quant_mode']


def load_and_filter_raw_data(filepath: str) -> pd.DataFrame:
    """Applies row-level data quality exclusion criteria."""
    raw_df = pd.read_csv(filepath, parse_dates=['timestamp'])
    # Exclude exact collector duplicates and aborted 0-token health-check pings
    df_valid = raw_df.drop_duplicates().copy()
    df_valid = df_valid[df_valid['prompt_tokens'] > 0].sort_values('timestamp').reset_index(drop=True)
    return df_valid


def build_hgb_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), FEATURE_COLS_NUM),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), FEATURE_COLS_CAT)
        ]
    )
    return Pipeline([
        ('sanitizer', TelemetrySanitizer()),
        ('featurizer', KVCacheFeatureEngineer(rolling_window=6)),
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.2,
            class_weight='balanced', random_state=42
        ))
    ])


def optimize_cost_threshold(y_true: np.ndarray, probs: np.ndarray, fn_cost=45.0, fp_cost=10.0) -> float:
    thresholds = np.linspace(0.05, 0.90, 86)
    costs = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        costs.append((fn * fn_cost + fp * fp_cost) / len(y_true))
    return float(thresholds[np.argmin(costs)])


if __name__ == '__main__':
    df_valid = load_and_filter_raw_data('data/llm_inference_telemetry_raw.csv')
    split_idx = int(len(df_valid) * 0.80)
    train_df, test_df = df_valid.iloc[:split_idx], df_valid.iloc[split_idx:]
    y_train, y_test = train_df['preemption_event'].values, test_df['preemption_event'].values

    pipeline = build_hgb_pipeline()
    calibrated_model = CalibratedClassifierCV(pipeline, method='isotonic', cv=3)
    calibrated_model.fit(train_df, y_train)

    test_probs = calibrated_model.predict_proba(test_df)[:, 1]
    opt_tau = optimize_cost_threshold(y_test, test_probs)
    test_preds = (test_probs >= opt_tau).astype(int)

    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    print(f"Test PR-AUC: {auc(rec, prec):.4f} | Brier Score: {brier_score_loss(y_test, test_probs):.4f}")
    print(f"Optimal Cost-Aware Threshold: {opt_tau:.2f}")
    print(classification_report(y_test, test_preds, target_names=['Stable (0)', 'KV Preemption (1)']))
