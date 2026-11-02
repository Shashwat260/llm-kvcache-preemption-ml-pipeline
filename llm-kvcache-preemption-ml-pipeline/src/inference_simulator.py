import pandas as pd
from pipeline_trainer import load_and_filter_raw_data, build_hgb_pipeline, optimize_cost_threshold
from sklearn.calibration import CalibratedClassifierCV


def run_routing_simulation():
    df_valid = load_and_filter_raw_data('data/llm_inference_telemetry_raw.csv')
    split_idx = int(len(df_valid) * 0.80)
    train_df, test_df = df_valid.iloc[:split_idx], df_valid.iloc[split_idx:].copy()

    base_pipe = build_hgb_pipeline()
    cal_pipe = CalibratedClassifierCV(base_pipe, method='isotonic', cv=3)
    cal_pipe.fit(train_df, train_df['preemption_event'].values)

    probs = cal_pipe.predict_proba(test_df)[:, 1]
    tau = optimize_cost_threshold(test_df['preemption_event'].values, probs)

    test_df['preemption_risk_prob'] = probs.round(4)
    test_df['scheduler_action'] = [
        'REROUTE_SECONDARY_CLUSTER' if p >= tau else 'ADMIT_LOCAL_KV_POOL'
        for p in probs
    ]

    sample_alerts = test_df[test_df['scheduler_action'] == 'REROUTE_SECONDARY_CLUSTER'].head(5)
    cols = ['timestamp', 'cluster_id', 'model_tier', 'batch_size', 'prompt_tokens', 'kv_cache_util_pct', 'preemption_risk_prob', 'scheduler_action']
    print("=== Proactive KV-Cache Admission Control Decisions (Sample Alerts) ===")
    print(sample_alerts[cols].to_string(index=False))


if __name__ == '__main__':
    run_routing_simulation()
