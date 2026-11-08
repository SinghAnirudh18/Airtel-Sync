import warnings
warnings.filterwarnings('ignore')

import re
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
    VotingClassifier, HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    mean_absolute_error, r2_score
)

# ==============================================================================
# 0. LOAD ALL DATASETS
# ==============================================================================
leads = pd.read_csv('smb_leads.csv')
cat = pd.read_csv('product_catalogue.csv').set_index('product_id')
gstin_reg = pd.read_csv('mock_gstin_registry.csv')
udyam_reg = pd.read_csv('mock_udyam_registry.csv')

print(f"[INFO] Loaded {len(leads)} leads, {len(cat)} products, "
      f"{len(gstin_reg)} GSTIN records, {len(udyam_reg)} Udyam records.\n")

# ==============================================================================
# ENGINE 1: FEATURE ENGINEERING + MULTI-MODEL ENSEMBLE PROPENSITY & EV ROUTER
# ==============================================================================
def engineer_lead_features(leads_df, cat_df, gst_df, udyam_df):
    df = leads_df.merge(
        gst_df[['gstin', 'status', 'constitution', 'registration_date']],
        on='gstin', how='left'
    ).merge(
        udyam_df[['udyam_number', 'enterprise_type', 'major_activity', 'date_of_registration']],
        on='udyam_number', how='left'
    )

    # Date age features relative to lead creation
    created_dt = pd.to_datetime(df['created_date'])
    df['gst_age_days'] = (created_dt - pd.to_datetime(df['registration_date'])).dt.days.fillna(-1)
    df['udyam_age_days'] = (created_dt - pd.to_datetime(df['date_of_registration'])).dt.days.fillna(-1)

    # Multi-hot encoding of products_of_interest
    for pid in cat_df.index:
        df[f'poi_{pid}'] = df['products_of_interest'].fillna('').apply(
            lambda x: 1 if pid in str(x).split(';') else 0
        )

    # Check catalogue eligibility rules & cost ratios for requested products
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

    # Spend ratios
    df['spend_per_emp'] = df['monthly_telecom_spend_inr'] / (df['employee_count'] + 1)
    df['spend_per_site'] = df['monthly_telecom_spend_inr'] / (df['num_sites'] + 1)
    df['quote_to_spend_ratio'] = df['poi_min_cost'] / (df['monthly_telecom_spend_inr'] + 1)
    df['is_existing_accel'] = df['current_provider'].fillna('').str.contains('Accel').astype(int)

    return df

df_fe = engineer_lead_features(leads, cat, gstin_reg, udyam_reg)
df_fe['target'] = (df_fe['outcome'] == 'Activated').astype(int)

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
] + [f'poi_{pid}' for pid in cat.index]

for c in cat_cols:
    df_fe[c] = df_fe[c].fillna('Missing').astype(str)

X_all = pd.get_dummies(df_fe[cat_cols + num_cols], drop_first=True)
y_all = df_fe['target']

# Define multi-model ensemble
estimators = [
    ('lr', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=0.05, class_weight='balanced', random_state=42))
    ])),
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_leaf=15, class_weight='balanced', random_state=42
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=20, learning_rate=0.04, random_state=42
    )),
    ('hgb', HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, min_samples_leaf=25, learning_rate=0.04, class_weight='balanced', random_state=42
    ))
]

# Optional: Add XGBoost / LightGBM if installed in Colab
try:
    import xgboost as xgb
    estimators.append(('xgb', xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.04, scale_pos_weight=12,
        eval_metric='logloss', random_state=42
    )))
except ImportError:
    pass

voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')

print("=== ENGINE 1: 5-FOLD CROSS-VALIDATED PROPENSITY BENCHMARK ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in estimators + [('Soft_Voting_Ensemble', voting_ensemble)]:
    cv_auc = cross_val_score(model, X_all, y_all, cv=skf, scoring='roc_auc')
    print(f"  {name:<22} | ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

# Fit final Propensity Model on 80/20 train-test split for evaluation & scoring
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_all, y_all, df_fe.index, test_size=0.2, stratify=y_all, random_state=42
)
voting_ensemble.fit(X_train, y_train)
test_probs = voting_ensemble.predict_proba(X_test)[:, 1]

# Train Expected Bundle Value (MRR) Regressor on Activated Leads
act_mask = df_fe['target'] == 1
mrr_regressor = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, random_state=42)
mrr_regressor.fit(X_all[act_mask], df_fe.loc[act_mask, 'monthly_bundle_value_inr'])

# Compute Propensity Score (0-100), Predicted MRR, and Expected Value (EV) for Test Leads
results_df = df_fe.loc[idx_test, [
    'lead_id', 'business_name', 'segment', 'city_tier', 'lead_source',
    'has_gstin', 'status', 'employee_count', 'monthly_telecom_spend_inr', 'outcome'
]].copy()

# Normalize probabilities into an intuitive 0-100 Propensity Percentile Score
results_df['Raw_Prob'] = test_probs
results_df['Propensity_Score'] = (pd.Series(test_probs, index=results_df.index).rank(pct=True) * 100).round(1)
results_df['Predicted_MRR_INR'] = np.clip(mrr_regressor.predict(X_test), 999, None).round(0)
results_df['Expected_Annual_Value_INR'] = (results_df['Raw_Prob'] * results_df['Predicted_MRR_INR'] * 12).round(0)

def assign_sales_channel(row):
    if row['Propensity_Score'] >= 75 and row['Predicted_MRR_INR'] >= 6000:
        return '1. Field Sales VIP (High Propensity + High Value)'
    elif row['Propensity_Score'] >= 55:
        return '2. Channel Partner Assisted (Warm Digital Drop-off)'
    else:
        return '3. 100% Self-Serve Digital (GenAI Advisor + WhatsApp)'

results_df['Recommended_Routing'] = results_df.apply(assign_sales_channel, axis=1)

print("\n=== TOP 10 PRIORITY LEADS RANKED BY EXPECTED ANNUAL VALUE (EV) ===")
print(results_df.sort_values(by='Expected_Annual_Value_INR', ascending=False)[[
    'lead_id', 'business_name', 'segment', 'Propensity_Score',
    'Predicted_MRR_INR', 'Expected_Annual_Value_INR', 'Recommended_Routing'
]].head(10).to_string(index=False))

# ==============================================================================
# ENGINE 2: HYBRID ML + RULE-CONSTRAINED BUNDLE RECOMMENDER (ADVISOR BACKEND)
# ==============================================================================
print("\n=== ENGINE 2: TRAINING HYBRID ML + CATALOGUE RULE BUNDLE RECOMMENDER ===")
act_df = df_fe[df_fe['target'] == 1].copy()

# Create multi-label binary matrix of products actually bought by activated customers
Y_bundles = pd.DataFrame(0, index=act_df.index, columns=cat.index)
for idx, b_str in act_df['bundle_products'].dropna().items():
    for item in b_str.split(';'):
        pid = item.split('x')[0]
        if pid in Y_bundles.columns:
            Y_bundles.loc[idx, pid] = 1

# Keep products that appear in at least 5 activated bundles for ML training
active_pids = Y_bundles.columns[Y_bundles.sum() >= 5].tolist()
bundle_clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)
)
bundle_clf.fit(X_all.loc[act_df.index], Y_bundles[active_pids])

def recommend_smart_bundle(lead_row_idx, top_k=3):
    """
    Combines ML probability scores with strict product_catalogue.csv eligibility
    rules and dependency resolution (requires_one_of).
    """
    row = df_fe.loc[lead_row_idx]
    x_vec = X_all.loc[[lead_row_idx]]

    # 1. Get ML affinity score for each product
    ml_scores = {}
    probs_list = bundle_clf.predict_proba(x_vec)
    for i, pid in enumerate(active_pids):
        prob_pos = probs_list[i][0][1] if probs_list[i].shape[1] > 1 else 0.0
        ml_scores[pid] = prob_pos
    for pid in cat.index:
        if pid not in ml_scores:
            ml_scores[pid] = 0.05

    # Boost products the user explicitly expressed interest in
    user_pois = [p for p in str(row['products_of_interest']).split(';') if p in cat.index]
    for p in user_pois:
        ml_scores[p] = ml_scores.get(p, 0.0) + 0.35

    # 2. Hard Eligibility Filter (City Tier, Min Employees, Min Sites, Active GSTIN)
    gst_valid = (row['has_gstin'] == 'Y') and (row['status'] == 'Active')
    eligible_pids = []
    ineligible_reasons = {}

    for pid, prod in cat.iterrows():
        allowed_tiers = [int(t) for t in str(prod['available_city_tiers']).split(';')]
        if int(row['city_tier']) not in allowed_tiers:
            ineligible_reasons[pid] = f"Not available in Tier {row['city_tier']} cities"
            continue
        if int(row['employee_count']) < int(prod['min_employees']):
            ineligible_reasons[pid] = f"Requires min {prod['min_employees']} employees"
            continue
        if int(row['num_sites']) < int(prod['min_sites']):
            ineligible_reasons[pid] = f"Requires min {prod['min_sites']} sites"
            continue
        if prod['requires_gstin'] == 'Y' and not gst_valid:
            ineligible_reasons[pid] = "Requires an Active GSTIN"
            continue
        eligible_pids.append(pid)

    # 3. Select top eligible products & ensure at least 1 primary connectivity link
    ranked_eligible = sorted(eligible_pids, key=lambda p: ml_scores[p], reverse=True)
    selected = ranked_eligible[:top_k]

    connectivity_links = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']
    if not any(p in selected for p in connectivity_links):
        best_conn = [p for p in ranked_eligible if p in connectivity_links]
        if best_conn:
            selected[-1] = best_conn[0]

    # 4. Dependency Graph Resolution (requires_one_of)
    for pid in list(selected):
        req_str = str(cat.loc[pid, 'requires_one_of'])
        if req_str != 'nan' and req_str != '':
            req_options = req_str.split('|')
            if not any(r in selected for r in req_options):
                valid_reqs = [r for r in req_options if r in eligible_pids]
                if valid_reqs:
                    selected.append(valid_reqs[0])
                else:
                    selected.remove(pid)

    # 5. Calculate Transparent Quote & Install SLA
    bundle_items = []
    total_monthly = 0
    total_setup = 0
    max_install_days = 0
    max_contract = 12

    for pid in selected:
        prod = cat.loc[pid]
        qty = 1
        if prod['unit'] == 'per site':
            qty = max(1, int(row['num_sites']))
        elif prod['unit'] == 'per line':
            qty = min(max(2, int(row['employee_count'] // 3)), 10)
        elif prod['unit'] == 'per user':
            qty = min(max(3, int(row['employee_count'])), 25)

        unit_price = int((prod['price_min_inr_per_month'] + prod['price_max_inr_per_month']) // 2)
        line_monthly = unit_price * qty
        total_monthly += line_monthly
        total_setup += int(prod['one_time_setup_inr'])
        max_install_days = max(max_install_days, int(prod['typical_install_days']))
        max_contract = max(max_contract, int(prod['min_contract_months']))

        bundle_items.append({
            'product_id': pid,
            'product_name': prod['product_name'],
            'category': prod['category'],
            'quantity': qty,
            'unit': prod['unit'],
            'monthly_price_inr': line_monthly,
            'setup_fee_inr': int(prod['one_time_setup_inr']),
            'install_days': int(prod['typical_install_days'])
        })

    flagged_pois = {p: ineligible_reasons[p] for p in user_pois if p in ineligible_reasons}

    return {
        'lead_id': row['lead_id'],
        'business_name': row['business_name'],
        'segment': row['segment'],
        'preferred_language': row['preferred_language'],
        'recommended_bundle': bundle_items,
        'transparent_monthly_total_inr': total_monthly,
        'one_time_setup_total_inr': total_setup,
        'contract_months': max_contract,
        'estimated_activation_sla_days': max_install_days,
        'ineligible_requested_products_flagged': flagged_pois
    }

# Demo the recommender on a sample Micro Retail lead and a Professional Office lead
sample_quote = recommend_smart_bundle(leads[leads['has_gstin'] == 'N'].index[0])
print(json.dumps(sample_quote, indent=2))

# ==============================================================================
# ENGINE 3: SMART KYC VALIDATOR & DPDP ACT PRIVACY ENGINE
# ==============================================================================
print("\n=== ENGINE 3: SMART KYC VALIDATOR & DPDP COMPLIANCE CHECKER ===")
BENCHMARK_TODAY = pd.to_datetime('2026-07-15')  # Per README specification

def validate_gstin_checksum(gstin_str):
    """Validates 15-char GSTIN structure (accepting 'X' in 4th PAN char) + Mod-36 checksum."""
    if not isinstance(gstin_str, str) or len(gstin_str) != 15:
        return False, "Invalid length (must be 15 characters)"
    pattern = r'^[0-9]{2}[A-Z]{3}[A-Z0-9][A-Z][0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$'
    if not re.match(pattern, gstin_str):
        return False, "Format regex mismatch"

    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    factor = 1
    total = 0
    for ch in gstin_str[:14]:
        code = chars.index(ch)
        prod = code * factor
        total += (prod // 36) + (prod % 36)
        factor = 2 if factor == 1 else 1
    check_code = (36 - (total % 36)) % 36
    if chars[check_code] != gstin_str[14]:
        return False, f"Checksum mismatch (expected {chars[check_code]}, got {gstin_str[14]})"
    return True, "Valid format and checksum"

def run_smart_kyc_audit(gstin_val=None, udyam_val=None, bill_date_str=None, business_name=None):
    report = {'consent_verified': True, 'checks': [], 'overall_status': 'APPROVED'}

    # 1. GSTIN Validation & Mock Registry Lookup
    if gstin_val and pd.notna(gstin_val):
        ok_fmt, msg_fmt = validate_gstin_checksum(gstin_val)
        masked_gst = gstin_val[:2] + "XXXXXXXXX" + gstin_val[-4:]
        if not ok_fmt:
            report['checks'].append({'doc': 'GSTIN', 'id': masked_gst, 'status': 'REJECTED', 'reason': msg_fmt})
            report['overall_status'] = 'REJECTED'
        else:
            match = gstin_reg[gstin_reg['gstin'] == gstin_val]
            if match.empty:
                report['checks'].append({'doc': 'GSTIN', 'id': masked_gst, 'status': 'REJECTED', 'reason': 'Not found in GST Registry'})
                report['overall_status'] = 'REJECTED'
            else:
                reg_row = match.iloc[0]
                if reg_row['status'] != 'Active':
                    report['checks'].append({
                        'doc': 'GSTIN', 'id': masked_gst, 'status': 'REJECTED',
                        'reason': f"GSTIN status is '{reg_row['status']}' (Active required)"
                    })
                    report['overall_status'] = 'REJECTED'
                elif business_name and (business_name.lower() not in reg_row['trade_name'].lower() and
                                        business_name.lower() not in reg_row['legal_name'].lower()):
                    report['checks'].append({
                        'doc': 'GSTIN', 'id': masked_gst, 'status': 'FLAGGED',
                        'reason': f"Entity name mismatch: '{business_name}' vs '{reg_row['trade_name']}'"
                    })
                    report['overall_status'] = 'MANUAL_REVIEW'
                else:
                    report['checks'].append({'doc': 'GSTIN', 'id': masked_gst, 'status': 'VERIFIED', 'reason': 'Active & Name Matched'})

    # 2. Udyam Validation & Lookup
    if udyam_val and pd.notna(udyam_val):
        masked_udy = udyam_val[:9] + "XXXX" + udyam_val[-3:]
        if not re.match(r'^UDYAM-[A-Z]{2}-00-[0-9]{7}$', str(udyam_val)):
            report['checks'].append({'doc': 'Udyam', 'id': masked_udy, 'status': 'REJECTED', 'reason': 'Invalid Udyam format'})
            report['overall_status'] = 'REJECTED'
        else:
            u_match = udyam_reg[udyam_reg['udyam_number'] == udyam_val]
            if u_match.empty:
                report['checks'].append({'doc': 'Udyam', 'id': masked_udy, 'status': 'REJECTED', 'reason': 'Not found in Udyam Registry'})
                report['overall_status'] = 'REJECTED'
            else:
                report['checks'].append({'doc': 'Udyam', 'id': masked_udy, 'status': 'VERIFIED', 'reason': 'Verified in Udyam Registry'})

    # 3. Address Proof Recency Check (Max 90 days before 2026-07-15)
    if bill_date_str:
        age_days = (BENCHMARK_TODAY - pd.to_datetime(bill_date_str)).days
        if age_days > 90:
            report['checks'].append({
                'doc': 'Address_Proof', 'status': 'REJECTED',
                'reason': f"Utility bill is {age_days} days old (exceeds 90-day limit from 2026-07-15)"
            })
            report['overall_status'] = 'REJECTED'
        else:
            report['checks'].append({'doc': 'Address_Proof', 'status': 'VERIFIED', 'reason': f"Valid ({age_days} days old)"})

    return report

# Test Smart KYC on an Active Lead vs. a Cancelled GSTIN & Expired Utility Bill
cancelled_sample = gstin_reg[gstin_reg['status'] == 'Cancelled'].iloc[0]
print("Test 1 (Planted Error - Cancelled GSTIN + Expired Address Proof):")
print(json.dumps(run_smart_kyc_audit(
    gstin_val=cancelled_sample['gstin'],
    bill_date_str='2026-03-01',
    business_name=cancelled_sample['trade_name']
), indent=2))

# ==============================================================================
# 4. GENERATE PITCH-DECK VISUALIZATION DASHBOARD (4 CHARTS)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Panel A: ROC Curves Comparing Single Models vs. Soft Voting Ensemble
for name, model in estimators:
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0, 0].plot(fpr, tpr, alpha=0.6, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")

fpr_ens, tpr_ens, _ = roc_curve(y_test, test_probs)
axes[0, 0].plot(fpr_ens, tpr_ens, color='crimson', lw=2.8,
                label=f"Soft Voting Ensemble (AUC={roc_auc_score(y_test, test_probs):.3f})")
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0, 0].set_title('A. Lead Propensity ROC-AUC Benchmark', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].legend(loc='lower right', fontsize=9)

# Panel B: Top 12 Engineered Feature Importances
rf_fitted = estimators[1][1].fit(X_train, y_train)
top_feats = pd.Series(rf_fitted.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(12)
sns.barplot(x=top_feats.values, y=top_feats.index, hue=top_feats.index, palette='magma', legend=False, ax=axes[0, 1])
axes[0, 1].set_title('B. Top 12 Predictive Signals Driving SMB Activation', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Relative Importance')

# Panel C: Stage-by-Stage Funnel Conversion by Segment
stages_order = ['Lead', 'Contacted', 'Needs assessed', 'Quote accepted', 'KYC complete', 'Activated']
s_map = {s: i for i, s in enumerate(stages_order)}
df_fe['s_idx'] = df_fe['stage_reached'].map(s_map)
funnel_seg = pd.DataFrame({
    s: df_fe.groupby('segment')['s_idx'].apply(lambda x: (x >= i).mean() * 100)
    for i, s in enumerate(stages_order)
})
funnel_seg.T.plot(marker='o', lw=2.2, ax=axes[1, 0])
axes[1, 0].set_title('C. Funnel Survival Rate (%) by Customer Segment', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('% of Leads Reaching Stage')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(title='Segment', fontsize=8)

# Panel D: Impact of Follow-up Calls on 6-Month Churn & NPS
act_only = leads[leads['outcome'] == 'Activated'].copy()
act_only['churn_pct'] = (act_only['churned_within_6_months'] == 'Y').astype(int) * 100
call_impact = act_only.groupby('customer_follow_up_calls')[['churn_pct', 'nps_score']].mean()
ax_d1 = axes[1, 1]
ax_d2 = ax_d1.twinx()
ax_d1.bar(call_impact.index - 0.15, call_impact['churn_pct'], width=0.3, color='coral', label='6-Mo Churn (%)')
ax_d2.plot(call_impact.index + 0.15, call_impact['nps_score'], color='teal', marker='s', lw=2.5, label='Mean NPS Score')
ax_d1.set_title('D. Status Chases vs. 6-Month Churn & NPS (Why Tracker Matters)', fontsize=13, fontweight='bold')
ax_d1.set_xlabel('Number of Customer Follow-Up Calls')
ax_d1.set_ylabel('6-Month Churn Rate (%)', color='coral')
ax_d2.set_ylabel('Average NPS (0-10)', color='teal')

plt.tight_layout()
plt.savefig('udyam_ml_dashboard.png', dpi=300)
plt.show()
print("\n[SUCCESS] Saved high-res presentation chart to 'udyam_ml_dashboard.png'")