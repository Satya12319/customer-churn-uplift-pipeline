import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def simulate_causal_customer_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Simulates customer retention data with realistic causal treatment effects across
    the 4 Uplift Quadrants (Persuadables, Sure Things, Lost Causes, Sleeping Dogs).
    """
    np.random.seed(seed)
    days_since_login = np.random.randint(1, 30, n_customers)
    support_tickets = np.random.randint(0, 5, n_customers)
    received_offer = np.random.binomial(1, 0.5, n_customers)

    # 1. Baseline retention probability (without treatment)
    # Customers who logged in recently and have fewer support tickets are more likely to stay
    base_logit = 1.2 - (0.08 * days_since_login) - (0.35 * support_tickets)
    base_prob = 1 / (1 + np.exp(-base_logit))

    # 2. True Causal Treatment Effect (Uplift)
    # - Persuadables (moderate login gap, low tickets): Strong positive uplift (+25% to +35%)
    # - Sleeping Dogs (frustrated users with >=3 support tickets): Negative uplift (-20% when spammed)
    # - Sure Things / Lost Causes: Near-zero uplift
    treatment_effect = np.where(
        (support_tickets >= 3),
        -0.22,  # Sleeping Dogs: Poking frustrated customers triggers churn
        np.where(
            (days_since_login >= 7) & (days_since_login <= 20) & (support_tickets <= 1),
            0.30,   # Persuadables: Ideal retention offer targets
            0.02    # Sure Things & Lost Causes: Minimal impact from discount
        )
    )

    # Final retention probability combining baseline + causal uplift if treated
    final_prob = np.clip(base_prob + (received_offer * treatment_effect), 0.02, 0.98)
    retained = np.random.binomial(1, final_prob)

    return pd.DataFrame({
        'customer_id': range(n_customers),
        'days_since_login': days_since_login,
        'support_tickets': support_tickets,
        'received_offer': received_offer,
        'retained': retained
    })


def evaluate_uplift_metrics(y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray) -> dict:
    """
    Calculates out-of-sample Uplift Evaluation Metrics:
    - AUUC (Area Under the Uplift Curve)
    - Qini Coefficient (Normalized Area Above Random Targeting)
    - Uplift @ Top 20% (Incremental retention rate in top deciles)
    """
    eval_df = pd.DataFrame({
        'y': y_true,
        'uplift': uplift_pred,
        't': treatment
    }).sort_values('uplift', ascending=False).reset_index(drop=True)

    eval_df['n_t'] = eval_df['t'].cumsum()
    eval_df['n_c'] = (1 - eval_df['t']).cumsum()
    eval_df['y_t'] = (eval_df['y'] * eval_df['t']).cumsum()
    eval_df['y_c'] = (eval_df['y'] * (1 - eval_df['t'])).cumsum()

    # Calculate cumulative Qini curve gains
    eval_df['qini_curve'] = eval_df['y_t'] - (
        eval_df['y_c'] * (eval_df['n_t'] / np.maximum(eval_df['n_c'], 1))
    )

    total_qini_gain = eval_df['qini_curve'].iloc[-1]
    random_curve = np.linspace(0, total_qini_gain, len(eval_df))

    # Area under curves using trapezoidal integration
    auuc_score = float(np.trapezoid(eval_df['qini_curve'] - random_curve) / len(eval_df))

    # Theoretical practical maximum curve for normalization
    perfect_df = eval_df.sort_values(by=['y', 't'], ascending=[False, False]).reset_index(drop=True)
    perfect_df['n_t'] = perfect_df['t'].cumsum()
    perfect_df['n_c'] = (1 - perfect_df['t']).cumsum()
    perfect_df['y_t'] = (perfect_df['y'] * perfect_df['t']).cumsum()
    perfect_df['y_c'] = (perfect_df['y'] * (1 - perfect_df['t'])).cumsum()
    perfect_df['qini_perfect'] = perfect_df['y_t'] - (
        perfect_df['y_c'] * (perfect_df['n_t'] / np.maximum(perfect_df['n_c'], 1))
    )

    area_model = np.trapezoid(eval_df['qini_curve'] - random_curve)
    area_perfect = np.trapezoid(perfect_df['qini_perfect'] - random_curve)
    qini_coeff = float(area_model / area_perfect) if area_perfect != 0 else 0.0

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


def run_pipeline():
    # 1. Load Simulated Causal Dataset
    df = simulate_causal_customer_data(n_customers=5000, seed=42)
    features = ['days_since_login', 'support_tickets']

    # 2. Proper Stratified Train/Test Split (80% Train, 20% Out-of-Sample Holdout)
    stratify_col = df['received_offer'].astype(str) + "_" + df['retained'].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=stratify_col
    )

    # 3. Split Training Data into Treatment and Control for T-Learner
    df_treated = train_df[train_df['received_offer'] == 1]
    df_control = train_df[train_df['received_offer'] == 0]

    # 4. Train Two XGBoost Estimators (Regularized to prevent overfitting)
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model_treated = XGBClassifier(**xgb_params)
    model_treated.fit(df_treated[features], df_treated['retained'])

    model_control = XGBClassifier(**xgb_params)
    model_control.fit(df_control[features], df_control['retained'])

    # 5. Evaluate Out-of-Sample Uplift Performance on Test Set
    test_prob_t = model_treated.predict_proba(test_df[features])[:, 1]
    test_prob_c = model_control.predict_proba(test_df[features])[:, 1]
    test_uplift = test_prob_t - test_prob_c

    metrics = evaluate_uplift_metrics(
        y_true=test_df['retained'].values,
        uplift_pred=test_uplift,
        treatment=test_df['received_offer'].values
    )

    # 6. Score Active Customers for Power BI Dashboard
    df_active = df.copy()
    prob_if_treated = model_treated.predict_proba(df_active[features])[:, 1]
    prob_if_control = model_control.predict_proba(df_active[features])[:, 1]

    df_active['baseline_stay_prob'] = prob_if_control
    df_active['uplift_score'] = prob_if_treated - prob_if_control

    # Log Quadrant Breakdown into Metrics Artifact
    metrics['total_customers_scored'] = int(len(df_active))
    metrics['target_persuadables_count'] = int((df_active['uplift_score'] > 0.05).sum())
    metrics['sleeping_dogs_count'] = int((df_active['uplift_score'] < -0.05).sum())

    # 7. Export Scored Data & Validation Metrics
    df_active.to_csv('scored_customers.csv', index=False)
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Pipeline executed successfully!")
    print(f"Out-of-Sample Evaluation Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    run_pipeline()