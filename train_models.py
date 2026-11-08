import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, VotingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

os.makedirs('models', exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD DATASETS
# -----------------------------------------------------------------------------
print("[1/5] Loading datasets...")
leads_df = pd.read_csv('smb_leads.csv')
cat_df = pd.read_csv('product_catalogue.csv').set_index('product_id')
gst_df = pd.read_csv('mock_gstin_registry.csv')
udyam_df = pd.read_csv('mock_udyam_registry.csv')

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print("[2/5] Engineering domain-specific telecom features...")
df = leads_df.merge(
    gst_df[['gstin', 'status', 'constitution', 'registration_date']],
    on='gstin', how='left'
).merge(
    udyam_df[['udyam_number', 'enterprise_type', 'major_activity', 'date_of_registration']],
    on='udyam_number', how='left'
)

created_dt = pd.to_datetime(df['created_date'])
df['gst_age_days'] = (created_dt - pd.to_datetime(df['registration_date'])).dt.days.fillna(-1)
df['udyam_age_days'] = (created_dt - pd.to_datetime(df['date_of_registration'])).dt.days.fillna(-1)

for pid in cat_df.index:
    df[f'poi_{pid}'] = df['products_of_interest'].fillna('').apply(
        lambda x: 1 if pid in str(x).split(';') else 0
    )

def extract_poi_signals(row):
    pois = [p for p in str(row['products_of_interest']).split(';') if p in cat_df.index]
    if not pois:
        return pd.Series([0, 0, 0, 0, 0, 0, 0],
                         index=['n_pois', 'poi_min_cost', 'poi_max_cost', 'poi_setup_cost',
                                'poi_max_install', 'poi_inelig_gst', 'poi_inelig_size'])
    sub = cat_df.loc[pois]
    inelig_gst = int(((sub['requires_gstin'] == 'Y') & (row['has_gstin'] != 'Y')).sum())
    inelig_size = int(((row['employee_count'] < sub['min_employees']) |
                       (row['num_sites'] < sub['min_sites'])).sum())
    return pd.Series([
        len(pois),
        sub['price_min_inr_per_month'].sum(),
        sub['price_max_inr_per_month'].sum(),
        sub['one_time_setup_inr'].sum(),
        sub['typical_install_days'].max(),
        inelig_gst,
        inelig_size
    ], index=['n_pois', 'poi_min_cost', 'poi_max_cost', 'poi_setup_cost',
              'poi_max_install', 'poi_inelig_gst', 'poi_inelig_size'])

poi_signals = df.apply(extract_poi_signals, axis=1)
df = pd.concat([df, poi_signals], axis=1)

df['spend_per_emp'] = df['monthly_telecom_spend_inr'] / (df['employee_count'] + 1)
df['spend_per_site'] = df['monthly_telecom_spend_inr'] / (df['num_sites'] + 1)
df['quote_to_spend_ratio'] = df['poi_min_cost'] / (df['monthly_telecom_spend_inr'] + 1)
df['is_existing_accel'] = df['current_provider'].fillna('').str.contains('Accel').astype(int)

df['target'] = (df['outcome'] == 'Activated').astype(int)

cat_cols = [
    'lead_source', 'business_type', 'segment', 'city', 'state', 'preferred_language',
    'annual_turnover_band', 'has_gstin', 'status', 'constitution', 'has_udyam',
    'enterprise_type', 'major_activity', 'current_internet', 'current_provider', 'contact_role'
]
num_cols = [
    'city_tier', 'employee_count', 'num_sites', 'monthly_telecom_spend_inr',
    'gst_age_days', 'udyam_age_days', 'n_pois', 'poi_min_cost', 'poi_max_cost',
    'poi_setup_cost', 'poi_max_install', 'poi_inelig_gst', 'poi_inelig_size',
    'spend_per_emp', 'spend_per_site', 'quote_to_spend_ratio', 'is_existing_accel'
] + [f'poi_{pid}' for pid in cat_df.index]

for c in cat_cols:
    df[c] = df[c].fillna('Missing').astype(str)

X_all = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# Clean feature names for tree-based algorithms & XGBoost
clean_cols = [re.sub(r'[\[\]<>, -]', '_', col) for col in X_all.columns]
X_all.columns = clean_cols
y_all = df['target']

# -----------------------------------------------------------------------------
# 3. TRAIN MODEL 1: PROPENSITY ENSEMBLE
# -----------------------------------------------------------------------------
print("[3/5] Fitting Model 1: Propensity Soft Voting Ensemble...")
estimators = [
    ('lr', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=0.05, class_weight='balanced', random_state=42))
    ])),
    ('rf', RandomForestClassifier(
        n_estimators=150, max_depth=7, min_samples_leaf=15, class_weight='balanced', random_state=42
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=20, learning_rate=0.04, random_state=42
    )),
    ('hgb', HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, min_samples_leaf=25, learning_rate=0.04, class_weight='balanced', random_state=42
    ))
]

try:
    import xgboost as xgb
    estimators.append(('xgb', xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.04, scale_pos_weight=12,
        eval_metric='logloss', random_state=42
    )))
except ImportError:
    pass

voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
voting_ensemble.fit(X_all, y_all)
ensemble_probs = voting_ensemble.predict_proba(X_all)[:, 1]
auc = roc_auc_score(y_all, ensemble_probs)
print(f"  -> Model 1 Propensity ROC-AUC: {auc:.4f}")

joblib.dump(voting_ensemble, 'models/propensity_ensemble_model.pkl')
print("  -> Saved 'models/propensity_ensemble_model.pkl'")

# -----------------------------------------------------------------------------
# 4. TRAIN MODEL 2: EXPECTED MRR REGRESSOR
# -----------------------------------------------------------------------------
print("[4/5] Fitting Model 2: Expected MRR Regressor...")
act_mask = df['target'] == 1
mrr_regressor = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, random_state=42)
mrr_regressor.fit(X_all[act_mask], df.loc[act_mask, 'monthly_bundle_value_inr'])

joblib.dump(mrr_regressor, 'models/mrr_regressor_model.pkl')
print("  -> Saved 'models/mrr_regressor_model.pkl'")

# -----------------------------------------------------------------------------
# 5. TRAIN MODEL 3: BUNDLE RECOMMENDER MULTI-OUTPUT CLASSIFIER
# -----------------------------------------------------------------------------
print("[5/5] Fitting Model 3: Multi-Output Bundle Recommender...")
act_df = df[df['target'] == 1].copy()
Y_bundles = pd.DataFrame(0, index=act_df.index, columns=cat_df.index)
for idx, b_str in act_df['bundle_products'].dropna().items():
    for item in b_str.split(';'):
        pid = item.split('x')[0]
        if pid in Y_bundles.columns:
            Y_bundles.loc[idx, pid] = 1

active_pids = Y_bundles.columns[Y_bundles.sum() >= 5].tolist()
bundle_clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)
)
bundle_clf.fit(X_all.loc[act_df.index], Y_bundles[active_pids])

joblib.dump(bundle_clf, 'models/bundle_recommender_model.pkl')
print("  -> Saved 'models/bundle_recommender_model.pkl'")

# -----------------------------------------------------------------------------
# SAVE PRE-SCORED LEADS TABLE FOR SUB-MILLISECOND API QUERIES
# -----------------------------------------------------------------------------
print("[INFO] Pre-scoring 5,000 leads using loaded Joblib ensemble & regressor...")
pred_probs = voting_ensemble.predict_proba(X_all)[:, 1]
pred_mrr = np.clip(mrr_regressor.predict(X_all), 999, None)

scored_df = leads_df[['lead_id', 'business_name', 'segment', 'city', 'city_tier', 'lead_source', 'employee_count', 'monthly_telecom_spend_inr', 'stage_reached', 'outcome', 'drop_reason']].copy()
scored_df['propensity_score'] = (pd.Series(pred_probs).rank(pct=True) * 100).round(1).values
scored_df['predicted_mrr'] = np.round(pred_mrr, 0).astype(int)
scored_df['expected_annual_value'] = np.round(pred_probs * pred_mrr * 12, 0).astype(int)

def assign_routing(row):
    if row['propensity_score'] >= 75 and row['predicted_mrr'] >= 6000:
        return 'Field Sales VIP (High EV)', 'vip'
    elif row['propensity_score'] >= 50 or str(row['stage_reached']) in ['Quote accepted', 'KYC complete']:
        return 'Partner Assisted (Warm Drop-off)', 'partner'
    else:
        return '100% Digital Self-Serve', 'digital'

routings = scored_df.apply(assign_routing, axis=1)
scored_df['routing'] = [r[0] for r in routings]
scored_df['route_badge'] = [r[1] for r in routings]

scored_df.to_csv('models/scored_leads.csv', index=False)
print("  -> Saved 'models/scored_leads.csv'")

# -----------------------------------------------------------------------------
# SAVE METADATA & FEATURE SCHEMA
# -----------------------------------------------------------------------------
metadata = {
    "feature_columns": list(X_all.columns),
    "active_pids": active_pids,
    "product_count": len(cat_df),
    "leads_count": len(leads_df),
    "propensity_auc": round(float(auc), 4),
    "saved_timestamp": pd.Timestamp.now().isoformat()
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n[COMPLETE] Successfully converted and saved all 3 models with joblib in 'models/'!")

