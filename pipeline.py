import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. Load Data (Simulated for this portfolio build)
np.random.seed(42)
n_customers = 5000
df = pd.DataFrame({
    'customer_id': range(n_customers),
    'days_since_login': np.random.randint(1, 30, n_customers),
    'support_tickets': np.random.randint(0, 5, n_customers),
    'received_offer': np.random.randint(0, 2, n_customers), # 1 = treated, 0 = control
    'retained': np.random.randint(0, 2, n_customers)
})

features = ['days_since_login', 'support_tickets']

# 2. Split into Treatment and Control
df_treated = df[df['received_offer'] == 1]
df_control = df[df['received_offer'] == 0]

# 3. Train the T-Learner (Two Models)
model_treated = XGBClassifier(eval_metric='logloss')
model_treated.fit(df_treated[features], df_treated['retained'])

model_control = XGBClassifier(eval_metric='logloss')
model_control.fit(df_control[features], df_control['retained'])

# 4. Score Current Active Customers
df_active = df.copy() 
prob_if_treated = model_treated.predict_proba(df_active[features])[:, 1]
prob_if_control = model_control.predict_proba(df_active[features])[:, 1]

df_active['baseline_stay_prob'] = prob_if_control
df_active['uplift_score'] = prob_if_treated - prob_if_control

# 5. Export for Power BI
df_active.to_csv('scored_customers.csv', index=False)
print("Pipeline executed. Data exported.")