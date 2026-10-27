import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def simulate_causal_customer_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Simulates enterprise SaaS/Telecom customer retention data with continuous
    financial features and causal treatment effects across the 4 Uplift Quadrants.
    """
    np.random.seed(seed)
    days_since_login = np.random.randint(1, 30, n_customers)
    support_tickets = np.random.randint(0, 5, n_customers)
    tenure_months = np.random.randint(1, 60, n_customers)
    monthly_charges = np.round(np.random.uniform(25.0, 150.0, n_customers), 2)
    received_offer = np.random.binomial(1, 0.5, n_customers)

    # 1. Baseline retention probability (without treatment)
    base_logit = (
        0.8
        - (0.07 * days_since_login)
        - (0.38 * support_tickets)
        + (0.015 * tenure_months)
        - (0.004 * monthly_charges)
    )
    base_prob = 1 / (1 + np.exp(-base_logit))

    # 2. True Causal Treatment Effect (Heterogeneous Uplift)
    # - Sleeping Dogs: High support tickets (>=3) get annoyed by marketing outreach
    # - Persuadables: Moderate login gap (6-22 days), low tickets (<=1), higher bill
    treatment_effect = np.where(
        support_tickets >= 3,
        -0.25 + np.random.normal(0, 0.03, n_customers),  # Sleeping Dogs
        np.where(
            (days_since_login >= 6) & (days_since_login <= 22) & (support_tickets <= 1),
            0.28 + (0.0005 * monthly_charges) + np.random.normal(0, 0.03, n_customers),  # Persuadables
            0.01 + np.random.normal(0, 0.02, n_customers)  # Sure Things & Lost Causes
        )
    )

    final_prob = np.clip(base_prob + (received_offer * treatment_effect), 0.02, 0.98)
    retained = np.random.binomial(1, final_prob)

    return pd.DataFrame({
        'customer_id': range(n_customers),
        'days_since_login': days_since_login,
        'support_tickets': support_tickets,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'received_offer': received_offer,
        'retained': retained
    })


def _compute_qini_curve(df_sorted: pd.DataFrame) -> np.ndarray:
    """Computes cumulative incremental gains along a ranked customer population."""
    n_t = df_sorted['t'].cumsum()
    n_c = (1 - df_sorted['t']).cumsum()
    y_t = (df_sorted['y'] * df_sorted['t']).cumsum()
    y_c = (df_sorted['y'] * (1 - df_sorted['t'])).cumsum()

    scale_factor = np.where(n_c > 0, n_t / n_c, 0.0)
    return (y_t - (y_c * scale_factor)).values


def evaluate_uplift_metrics(y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray) -> dict:
    """
    Calculates out-of-sample Causal Uplift Evaluation Metrics:
    - Qini Coefficient (Normalized ratio against theoretical optimum)
    - AUUC (Area Under the Uplift Curve above random baseline)
    - Uplift @ Top 20% (Net incremental retention rate in top two deciles)
    """
    eval_df = pd.DataFrame({
        'y': y_true,
        'uplift': uplift_pred,
        't': treatment
    }).sort_values('uplift', ascending=False).reset_index(drop=True)

    model_qini = _compute_qini_curve(eval_df)
    total_gain = model_qini[-1]
    random_curve = np.linspace(0, total_gain, len(eval_df))

    area_model = float(np.trapezoid(model_qini - random_curve))
    auuc_score = area_model / len(eval_df)

    # Theoretical Optimal Ranking for Perfect Qini Normalization
    perfect_df = eval_df.copy()
    perfect_df['perfect_rank'] = np.where(
        (perfect_df['t'] == 1) & (perfect_df['y'] == 1), 3,
        np.where(
            (perfect_df['t'] == 0) & (perfect_df['y'] == 0), 2,
            np.where((perfect_df['t'] == 1) & (perfect_df['y'] == 0), 1, 0)
        )
    )
    perfect_df = perfect_df.sort_values('perfect_rank', ascending=False).reset_index(drop=True)
    perfect_qini = _compute_qini_curve(perfect_df)

    area_perfect = float(np.trapezoid(perfect_qini - random_curve))
    qini_coeff = float(area_model / area_perfect) if area_perfect > 0 else 0.0

    # Uplift at Top 20% targeted population
    top_20 = eval_df.iloc[: int(len(eval_df) * 0.20)]
    tr_rate_top20 = top_20[top_20['t'] == 1]['y'].mean()
    ctrl_rate_top20 = top_20[top_20['t'] == 0]['y'].mean()
    uplift_at_20 = float(tr_rate_top20 - ctrl_rate_top20)

    return {
        'qini_coefficient': round(qini_coeff, 4),
        'auuc_score': round(auuc_score, 4),
        'uplift_at_top_20_pct': round(uplift_at_20, 4)
    }


def assign_quadrant(row: pd.Series) -> str:
    """Classifies customers into the 4 actionable retention quadrants."""
    if row['uplift_score'] >= 0.08:
        return 'Persuadables (Target)'
    elif row['uplift_score'] <= -0.05:
        return 'Sleeping Dogs (Do Not Disturb)'
    elif row['baseline_stay_prob'] >= 0.50:
        return 'Sure Things (Save Budget)'
    else:
        return 'Lost Causes (Zero ROI)'


def run_pipeline():
    # 1. Load Historical Dataset for Training & Holdout Validation
    df_historical = simulate_causal_customer_data(n_customers=5000, seed=42)
    features = ['days_since_login', 'support_tickets', 'tenure_months', 'monthly_charges']

    # 2. Stratified Train/Test Split (80% Train, 20% Out-of-Sample Holdout)
    stratify_col = df_historical['received_offer'].astype(str) + "_" + df_historical['retained'].astype(str)
    train_df, test_df = train_test_split(
        df_historical, test_size=0.20, random_state=42, stratify=stratify_col
    )

    # 3. Train Regularized T-Learner (Two Independent XGBoost Estimators)
    xgb_params = {
        'n_estimators': 120,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    df_treated = train_df[train_df['received_offer'] == 1]
    df_control = train_df[train_df['received_offer'] == 0]

    model_treated = XGBClassifier(**xgb_params)
    model_treated.fit(df_treated[features], df_treated['retained'])

    model_control = XGBClassifier(**xgb_params)
    model_control.fit(df_control[features], df_control['retained'])

    # 4. Evaluate Out-of-Sample Uplift Performance on Unseen Holdout Set
    test_prob_t = model_treated.predict_proba(test_df[features])[:, 1]
    test_prob_c = model_control.predict_proba(test_df[features])[:, 1]
    test_uplift = test_prob_t - test_prob_c

    metrics = evaluate_uplift_metrics(
        y_true=test_df['retained'].values,
        uplift_pred=test_uplift,
        treatment=test_df['received_offer'].values
    )

    # 5. Automated MLOps Quality Gate (Blocks bad models from production)
    if metrics['qini_coefficient'] < 0.05 or metrics['uplift_at_top_20_pct'] <= 0.0:
        raise RuntimeError(
            f"MLOps Quality Gate Failed! Qini ({metrics['qini_coefficient']}) below threshold."
        )

    # 6. Score Unseen Active Customer Cohort for Power BI (Zero Data Leakage)
    df_active = simulate_causal_customer_data(n_customers=5000, seed=101)
    prob_if_treated = model_treated.predict_proba(df_active[features])[:, 1]
    prob_if_control = model_control.predict_proba(df_active[features])[:, 1]

    df_active['baseline_stay_prob'] = np.round(prob_if_control, 6)
    df_active['uplift_score'] = np.round(prob_if_treated - prob_if_control, 6)
    df_active['retention_quadrant'] = df_active.apply(assign_quadrant, axis=1)

    # Calculate Expected Net Annual ROI assuming a $15 retention offer cost
    offer_cost = 15.0
    df_active['expected_net_roi'] = np.round(
        (df_active['uplift_score'] * df_active['monthly_charges'] * 12) - offer_cost, 2
    )

    # 7. Log Executive Business KPIs into Metrics Artifact
    persuadables = df_active[df_active['retention_quadrant'] == 'Persuadables (Target)']
    metrics['total_customers_scored'] = int(len(df_active))
    metrics['target_persuadables_count'] = int(len(persuadables))
    metrics['sleeping_dogs_count'] = int(
        (df_active['retention_quadrant'] == 'Sleeping Dogs (Do Not Disturb)').sum()
    )
    metrics['estimated_net_annual_roi_usd'] = round(float(persuadables['expected_net_roi'].sum()), 2)

    # 8. Export Production Artifacts
    df_active.to_csv('scored_customers.csv', index=False)
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Enterprise Pipeline executed & validated successfully!")
    print(f"Out-of-Sample Evaluation & ROI Metrics:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    run_pipeline()