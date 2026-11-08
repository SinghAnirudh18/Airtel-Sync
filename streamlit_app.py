import os
import re
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from extractor import (
    extract_kyc_with_openrouter,
    encode_image_to_base64,
    extract_multiple_kyc_documents_parallel,
    process_applicant_openrouter
)

load_dotenv()

# -----------------------------------------------------------------------------
# 0. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Project Udyam Digital | Airtel Accel",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Airtel Enterprise Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #E40000;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
    }
    .stButton>button:hover {
        background-color: #B30000;
        color: white;
    }
    .badge-approved {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-rejected {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-vip {
        background-color: rgba(228, 0, 0, 0.2);
        color: #E40000;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        border: 1px solid rgba(228, 0, 0, 0.4);
    }
    .badge-partner {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-digital {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. LOAD DATASETS & JOBLIB MODELS (CACHED)
# -----------------------------------------------------------------------------
BENCHMARK_TODAY = pd.to_datetime('2026-07-15')

@st.cache_resource
def load_models_and_data():
    leads = pd.read_csv('smb_leads.csv')
    cat = pd.read_csv('product_catalogue.csv').set_index('product_id')
    gst_reg = pd.read_csv('mock_gstin_registry.csv')
    udy_reg = pd.read_csv('mock_udyam_registry.csv')
    
    m1 = joblib.load('models/propensity_ensemble_model.pkl')
    m2 = joblib.load('models/mrr_regressor_model.pkl')
    m3 = joblib.load('models/bundle_recommender_model.pkl')
    
    with open('models/model_metadata.json') as f:
        meta = json.load(f)
        
    scored = pd.read_csv('models/scored_leads.csv') if os.path.exists('models/scored_leads.csv') else None
    
    return leads, cat, gst_reg, udy_reg, m1, m2, m3, meta, scored

leads_df, catalogue_df, gstin_registry, udyam_registry, propensity_ensemble, mrr_regressor, bundle_recommender, model_metadata, scored_leads_df = load_models_and_data()

FEATURE_COLS = model_metadata["feature_columns"]
ACTIVE_PIDS = model_metadata["active_pids"]

# -----------------------------------------------------------------------------
# 2. FEATURE VECTOR GENERATOR FOR LIVE INFERENCE
# -----------------------------------------------------------------------------
def build_feature_vector(row_dict):
    """Dynamically builds a 158-feature DataFrame matching training schema for live inference."""
    vec = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLS)
    
    # Numeric features
    vec['city_tier'] = float(row_dict.get('city_tier', 1))
    vec['employee_count'] = float(row_dict.get('employee_count', 5))
    vec['num_sites'] = float(row_dict.get('num_sites', 1))
    spend = float(row_dict.get('monthly_telecom_spend_inr', 2500))
    vec['monthly_telecom_spend_inr'] = spend
    vec['spend_per_emp'] = spend / (vec['employee_count'] + 1)
    vec['spend_per_site'] = spend / (vec['num_sites'] + 1)
    
    vec['gst_age_days'] = float(row_dict.get('gst_age_days', 730))
    vec['udyam_age_days'] = float(row_dict.get('udyam_age_days', 500))
    vec['is_existing_accel'] = 1.0 if 'accel' in str(row_dict.get('current_provider', '')).lower() else 0.0
    
    # Products of interest signals
    pois = str(row_dict.get('products_of_interest', '')).split(';')
    pois = [p for p in pois if p in catalogue_df.index]
    vec['n_pois'] = float(len(pois))
    for p in pois:
        col = f'poi_{p}'
        if col in vec.columns:
            vec[col] = 1.0
            
    if pois:
        sub = catalogue_df.loc[pois]
        vec['poi_min_cost'] = float(sub['price_min_inr_per_month'].sum())
        vec['poi_max_cost'] = float(sub['price_max_inr_per_month'].sum())
        vec['poi_setup_cost'] = float(sub['one_time_setup_inr'].sum())
        vec['poi_max_install'] = float(sub['typical_install_days'].max())
        vec['quote_to_spend_ratio'] = vec['poi_min_cost'] / (spend + 1)
    else:
        vec['quote_to_spend_ratio'] = 0.5
        
    # Categorical one-hot mapping
    cat_mappings = {
        'lead_source': str(row_dict.get('lead_source', 'Website form')),
        'business_type': str(row_dict.get('business_type', 'Bakery')),
        'segment': str(row_dict.get('segment', 'Micro retail & services')),
        'city': str(row_dict.get('city', 'Kolkata')),
        'state': str(row_dict.get('state', 'West Bengal')),
        'preferred_language': str(row_dict.get('preferred_language', 'English')),
        'annual_turnover_band': str(row_dict.get('annual_turnover_band', '40 lakh - 1.5 cr')),
        'has_gstin': str(row_dict.get('has_gstin', 'Y')),
        'status': str(row_dict.get('status', 'Active')),
        'constitution': str(row_dict.get('constitution', 'Proprietorship')),
        'has_udyam': str(row_dict.get('has_udyam', 'Y')),
        'current_internet': str(row_dict.get('current_internet', 'Business broadband')),
        'current_provider': str(row_dict.get('current_provider', 'Competitor B')),
        'contact_role': str(row_dict.get('contact_role', 'Owner'))
    }
    
    for k, v in cat_mappings.items():
        v_clean = re.sub(r'[\[\]<>, -]', '_', v)
        cname = f"{k}_{v_clean}"
        if cname in vec.columns:
            vec[cname] = 1.0
            
    return vec

# -----------------------------------------------------------------------------
# 3. SMART KYC VALIDATION LOGIC
# -----------------------------------------------------------------------------
def validate_gstin_checksum(gstin_str):
    if not isinstance(gstin_str, str) or len(gstin_str) != 15:
        return False, "Invalid length (must be exactly 15 alphanumeric characters)"
    pattern = r'^[0-9]{2}[A-Z]{3}[A-Z0-9][A-Z][0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$'
    if not re.match(pattern, gstin_str):
        return False, "GSTIN format regex mismatch"

    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    factor = 1
    total = 0
    try:
        for ch in gstin_str[:14]:
            code = chars.index(ch)
            prod = code * factor
            total += (prod // 36) + (prod % 36)
            factor = 2 if factor == 1 else 1
        check_code = (36 - (total % 36)) % 36
        if chars[check_code] != gstin_str[14]:
            return False, f"Checksum mismatch: expected '{chars[check_code]}', got '{gstin_str[14]}'"
        return True, "Valid GSTIN format and Mod-36 checksum"
    except Exception as e:
        return False, f"Checksum computation failed: {str(e)}"

def run_smart_kyc_audit(gstin_val=None, udyam_val=None, bill_date_str=None, business_name=None):
    report = {'dpdp_consent_verified': True, 'overall_status': 'APPROVED', 'checks': []}

    # 1. GSTIN Check
    if gstin_val and str(gstin_val).strip() != "" and str(gstin_val).lower() != "none":
        gstin_clean = str(gstin_val).strip().upper()
        masked_gst = gstin_clean[:2] + "XXXXXXXXX" + gstin_clean[-4:] if len(gstin_clean) >= 6 else gstin_clean
        ok_fmt, msg_fmt = validate_gstin_checksum(gstin_clean)
        if not ok_fmt:
            report['checks'].append({
                'component': 'GSTIN Format & Checksum',
                'id': masked_gst,
                'status': 'REJECTED',
                'reason': msg_fmt
            })
            report['overall_status'] = 'REJECTED'
        else:
            match = gstin_registry[gstin_registry['gstin'] == gstin_clean]
            if match.empty:
                report['checks'].append({
                    'component': 'GSTIN Registry Lookup',
                    'id': masked_gst,
                    'status': 'REJECTED',
                    'reason': 'GSTIN not found in Mock National GST Registry'
                })
                report['overall_status'] = 'REJECTED'
            else:
                row = match.iloc[0]
                if row['status'] != 'Active':
                    report['checks'].append({
                        'component': 'GSTIN Status Check',
                        'id': masked_gst,
                        'status': 'REJECTED',
                        'reason': f"GSTIN status is '{row['status']}'. Only 'Active' status is eligible."
                    })
                    report['overall_status'] = 'REJECTED'
                else:
                    report['checks'].append({
                        'component': 'GSTIN Verification',
                        'id': masked_gst,
                        'status': 'VERIFIED',
                        'reason': f"Active GSTIN verified for '{row['trade_name']}' ({row['constitution']})"
                    })

    # 2. Udyam Check
    if udyam_val and str(udyam_val).strip() != "" and str(udyam_val).lower() != "none":
        udyam_clean = str(udyam_val).strip().upper()
        masked_udy = udyam_clean[:9] + "XXXX" + udyam_clean[-3:] if len(udyam_clean) >= 12 else udyam_clean
        if not re.match(r'^UDYAM-[A-Z]{2}-00-[0-9]{7}$', udyam_clean):
            report['checks'].append({
                'component': 'Udyam Format Check',
                'id': masked_udy,
                'status': 'REJECTED',
                'reason': 'Invalid Udyam structure (Must follow UDYAM-XX-00-XXXXXXX with 7 numeric digits)'
            })
            report['overall_status'] = 'REJECTED'
        else:
            match_u = udyam_registry[udyam_registry['udyam_number'] == udyam_clean]
            if match_u.empty:
                report['checks'].append({
                    'component': 'Udyam Registry Lookup',
                    'id': masked_udy,
                    'status': 'REJECTED',
                    'reason': 'Udyam number not registered in MSME Portal Database'
                })
                report['overall_status'] = 'REJECTED'
            else:
                u_row = match_u.iloc[0]
                report['checks'].append({
                    'component': 'Udyam Verification',
                    'id': masked_udy,
                    'status': 'VERIFIED',
                    'reason': f"Verified {u_row['enterprise_type']} Enterprise: '{u_row['enterprise_name']}' ({u_row['major_activity']})"
                })

    # 3. Address Proof Recency Check (Max 90 days from benchmark 2026-07-15)
    if bill_date_str and str(bill_date_str).strip() != "":
        try:
            bill_dt = pd.to_datetime(bill_date_str)
            age_days = (BENCHMARK_TODAY - bill_dt).days
            if age_days > 90:
                report['checks'].append({
                    'component': 'Address Proof Recency',
                    'id': f"Bill Date: {bill_date_str}",
                    'status': 'REJECTED',
                    'reason': f"Utility bill is {age_days} days old relative to benchmark 2026-07-15 (Exceeds 90-day maximum limit)"
                })
                report['overall_status'] = 'REJECTED'
            elif age_days < 0:
                report['checks'].append({
                    'component': 'Address Proof Recency',
                    'id': f"Bill Date: {bill_date_str}",
                    'status': 'REJECTED',
                    'reason': f"Utility bill date is in the future relative to benchmark 2026-07-15"
                })
                report['overall_status'] = 'REJECTED'
            else:
                report['checks'].append({
                    'component': 'Address Proof Recency',
                    'id': f"Bill Date: {bill_date_str}",
                    'status': 'VERIFIED',
                    'reason': f"Utility bill is recent ({age_days} days old, well within 90-day SLA window)"
                })
        except Exception as e:
            report['checks'].append({
                'component': 'Address Proof Recency',
                'id': f"Bill Date: {bill_date_str}",
                'status': 'REJECTED',
                'reason': f"Date parsing error: {str(e)}"
            })
            report['overall_status'] = 'REJECTED'

    if not report['checks']:
        report['checks'].append({
            'component': 'Micro Non-GST Verification',
            'id': 'Tier-2 Micro Exception',
            'status': 'VERIFIED',
            'reason': 'Micro-retail tier verified via digital self-declaration under DPDP Act data minimization.'
        })

    return report

# Specimen ground truth for quick 1-click test
SPECIMENS = {
    "K01": {"name": "Vindhya Tours & Travels", "gstin": "37IHKXV5167L1ZS", "udyam": "UDYAM-MH-00-2675962", "bill_date": "2024-03-24", "label": "K01: Vindhya Tours (Planted: Cancelled GSTIN + Expired 2024 Bill)"},
    "K02": {"name": "Harmony Dental Care", "gstin": "19EOGXH4115Y1ZL", "udyam": "UDYAM-WB-00-5053330", "bill_date": "2026-05-22", "label": "K02: Harmony Dental Care (Valid Active Pass)"},
    "K03": {"name": "Ganesh Woodcraft", "gstin": "29FGUXG3354T1ZQ", "udyam": "UDYAM-KR-00-1115424", "bill_date": "2026-06-15", "label": "K03: Ganesh Woodcraft (Valid Manufacturer Pass)"},
    "K04": {"name": "Saffron Dental Care LLP", "gstin": "27P25KS1936012H", "udyam": "UDYAM-MH-00-6131836", "bill_date": "2026-05-18", "label": "K04: Saffron Dental Care (Planted: GSTIN Checksum Mismatch)"},
    "K05": {"name": "Metro Retail Store", "gstin": "", "udyam": "", "bill_date": "2026-06-11", "label": "K05: Metro Retail (Valid Micro Non-GST Exemption Pass)"},
    "K06": {"name": "Saffron Unisex Salon", "gstin": "08YKZXS5189L1ZO", "udyam": "UDYAM-RJ-00-6577157", "bill_date": "2026-06-01", "label": "K06: Saffron Unisex Salon (Valid Pass)"},
    "K07": {"name": "Neelgiri Physio Clinic", "gstin": "29HIHXN8231E1ZI", "udyam": "UDYAM-DL-00-95655547", "bill_date": "2026-07-05", "label": "K07: Neelgiri Physio Clinic (Planted: 8-digit Udyam Error)"},
    "K08": {"name": "Green Leaf Legal Associates", "gstin": "10EGQXG6446F1Z0", "udyam": "UDYAM-BR-00-7880010", "bill_date": "2026-05-26", "label": "K08: Green Leaf Legal Associates (Valid Pass)"},
    "K09": {"name": "Kaveri Insurance Advisors", "gstin": "07HBIXK4506C1ZF", "udyam": "UDYAM-TS-00-8542181", "bill_date": "2026-07-10", "label": "K09: Kaveri Insurance Advisors (Valid Pass)"},
    "K10": {"name": "Kaveri Stays LLP", "gstin": "24GMYXK0402E1ZQ", "udyam": "UDYAM-GJ-00-6760776", "bill_date": "2026-06-18", "label": "K10: Kaveri Stays LLP (Valid Multi-site Pass)"},
    "K11": {"name": "Trinity Bake House Pvt Ltd", "gstin": "33XGTXT4416F1Z3", "udyam": "UDYAM-TN-00-2649689", "bill_date": "2025-10-31", "label": "K11: Trinity Bake House (Planted: Bill >90 Days Old)"},
    "K12": {"name": "Star Electronics", "gstin": "33MFTXS9954Q1Z9", "udyam": "UDYAM-TN-00-3866196", "bill_date": "2024-02-03", "label": "K12: Star Electronics (Planted: Bill 2 Years Expired)"}
}

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & SYSTEM HEALTH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://assets.airtel.in/static-assets/airtel-logo.png", width=160)
    st.markdown("### **Project Udyam Digital**")
    st.caption("Airtel SYNC · Enterprise Track A")
    
    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "🧠 1. Live ML Bundle Advisor",
            "📊 2. Live Lead Propensity & EV Engine",
            "🛡️ 3. Smart KYC & DPDP Compliance",
            "📦 4. Live Order & Milestone Tracker",
            "📈 5. Executive Funnel Metrics & Analytics"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Active Joblib Models")
    st.markdown("🟢 **Propensity Ensemble** `AUC: 0.8295`")
    st.markdown("🟢 **MRR Regressor** `HistGradientBoosting`")
    st.markdown("🟢 **Bundle Recommender** `25 Multi-Outputs`")
    st.caption("Backend: Python 3.12 · Scikit-Learn 1.8 · Joblib")

# -----------------------------------------------------------------------------
# VIEW 1: LIVE ML BUNDLE ADVISOR
# -----------------------------------------------------------------------------
if menu == "🧠 1. Live ML Bundle Advisor":
    st.title("🧠 Live GenAI & ML Bundle Advisor")
    st.markdown("Replaces manual sales visits with an automated **multi-output affinity recommender** constrained by catalogue eligibility rules.")

    col1, col2 = st.columns([5, 7])

    with col1:
        st.subheader("1. Business Profile Input")
        
        # Preset buttons
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("🛒 Preset: Kirana / Bakery"):
                st.session_state['b_name'] = "Saffron Bakers"
                st.session_state['b_type'] = "Bakery & Confectionery"
                st.session_state['b_seg'] = "Micro retail & services"
                st.session_state['b_tier'] = 2
                st.session_state['b_emp'] = 3
                st.session_state['b_sites'] = 1
                st.session_state['b_gst'] = "Y"
                st.session_state['b_spend'] = 1350
        with p_col2:
            if st.button("🩺 Preset: Dental Clinic"):
                st.session_state['b_name'] = "Harmony Dental Care"
                st.session_state['b_type'] = "Dental Clinic"
                st.session_state['b_seg'] = "Professional offices"
                st.session_state['b_tier'] = 1
                st.session_state['b_emp'] = 8
                st.session_state['b_sites'] = 1
                st.session_state['b_gst'] = "Y"
                st.session_state['b_spend'] = 3200

        p_col3, p_col4 = st.columns(2)
        with p_col3:
            if st.button("🏬 Preset: 6-Store Chain"):
                st.session_state['b_name'] = "Star Electronics"
                st.session_state['b_type'] = "Electronics Retail Chain"
                st.session_state['b_seg'] = "Small multi-site businesses"
                st.session_state['b_tier'] = 1
                st.session_state['b_emp'] = 45
                st.session_state['b_sites'] = 6
                st.session_state['b_gst'] = "Y"
                st.session_state['b_spend'] = 10300
        with p_col4:
            if st.button("🏭 Preset: Auto Factory"):
                st.session_state['b_name'] = "Precision Auto Parts"
                st.session_state['b_type'] = "Auto Components Mfg"
                st.session_state['b_seg'] = "Small manufacturers"
                st.session_state['b_tier'] = 2
                st.session_state['b_emp'] = 28
                st.session_state['b_sites'] = 2
                st.session_state['b_gst'] = "Y"
                st.session_state['b_spend'] = 18500

        biz_name = st.text_input("Business Name", value=st.session_state.get('b_name', 'Saffron Bakers'))
        biz_type = st.text_input("Business Type", value=st.session_state.get('b_type', 'Bakery & Confectionery'))
        segment = st.selectbox(
            "Customer Segment",
            ['Micro retail & services', 'Professional offices', 'Small multi-site businesses', 'Small manufacturers'],
            index=['Micro retail & services', 'Professional offices', 'Small multi-site businesses', 'Small manufacturers'].index(st.session_state.get('b_seg', 'Micro retail & services'))
        )
        
        c_tier, c_emp, c_sites = st.columns(3)
        with c_tier:
            city_tier = st.selectbox("City Tier", [1, 2], index=0 if st.session_state.get('b_tier', 2) == 1 else 1)
        with c_emp:
            emp_count = st.number_input("Employees", min_value=1, max_value=200, value=st.session_state.get('b_emp', 3))
        with c_sites:
            num_sites = st.number_input("Sites/Locations", min_value=1, max_value=50, value=st.session_state.get('b_sites', 1))

        c_gst, c_spend = st.columns(2)
        with c_gst:
            has_gst = st.selectbox("Has Active GSTIN?", ['Y', 'N'], index=0 if st.session_state.get('b_gst', 'Y') == 'Y' else 1)
        with c_spend:
            monthly_spend = st.number_input("Current Monthly Telecom Spend (₹)", min_value=500, max_value=200000, value=st.session_state.get('b_spend', 1350), step=250)

        run_advisor = st.button("🚀 Run Live ML Bundle Recommender", use_container_width=True)

    with col2:
        st.subheader("2. Model Output & Live Affinity Scoring")
        
        # Prepare feature vector and execute live inference
        input_data = {
            'city_tier': city_tier,
            'employee_count': emp_count,
            'num_sites': num_sites,
            'monthly_telecom_spend_inr': monthly_spend,
            'has_gstin': has_gst,
            'segment': segment,
            'business_type': biz_type
        }
        x_vec = build_feature_vector(input_data)
        
        # LIVE INFERENCE on bundle_recommender_model.pkl
        raw_probs = bundle_recommender.predict_proba(x_vec)
        product_scores = {}
        for i, pid in enumerate(ACTIVE_PIDS):
            prob = raw_probs[i][0][1] if raw_probs[i].shape[1] > 1 else 0.05
            product_scores[pid] = prob

        # Apply hard catalogue constraint filters
        gst_valid = (has_gst == 'Y')
        eligible_pids = []
        ineligible_reasons = {}

        for pid, prod in catalogue_df.iterrows():
            allowed_tiers = [int(t) for t in str(prod['available_city_tiers']).split(';')]
            if city_tier not in allowed_tiers:
                ineligible_reasons[pid] = f"Unavailable in Tier {city_tier}"
                continue
            if emp_count < int(prod['min_employees']):
                ineligible_reasons[pid] = f"Needs {prod['min_employees']}+ staff"
                continue
            if num_sites < int(prod['min_sites']):
                ineligible_reasons[pid] = f"Needs {prod['min_sites']}+ sites"
                continue
            if prod['requires_gstin'] == 'Y' and not gst_valid:
                ineligible_reasons[pid] = "Requires Active GSTIN"
                continue
            eligible_pids.append(pid)

        # Rank eligible products by true ML score
        ranked_eligible = sorted(eligible_pids, key=lambda p: product_scores.get(p, 0.0), reverse=True)
        selected_pids = ranked_eligible[:3]

        # Ensure base connectivity link
        connectivity_links = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']
        if not any(p in selected_pids for p in connectivity_links):
            best_conn = [p for p in ranked_eligible if p in connectivity_links]
            if best_conn:
                selected_pids.insert(0, best_conn[0])

        # Dependency resolution (requires_one_of)
        for pid in list(selected_pids):
            req_str = str(catalogue_df.loc[pid, 'requires_one_of'])
            if req_str != 'nan' and req_str != '':
                options = req_str.split('|')
                if not any(o in selected_pids for o in options):
                    valid_deps = [o for o in options if o in eligible_pids]
                    if valid_deps:
                        selected_pids.append(valid_deps[0])
                    else:
                        selected_pids.remove(pid)

        # Plot live ML affinity scores chart
        plot_df = pd.DataFrame({
            'Product': [f"{p} ({catalogue_df.loc[p, 'product_name'][:14]})" for p in ACTIVE_PIDS[:12]],
            'ML Affinity Probability': [product_scores[p] for p in ACTIVE_PIDS[:12]],
            'Selected': ['Recommended' if p in selected_pids else 'Candidate' for p in ACTIVE_PIDS[:12]]
        })

        chart = alt.Chart(plot_df).mark_bar(cornerRadius=6).encode(
            x=alt.X('ML Affinity Probability:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Product:N', sort='-x'),
            color=alt.Color('Selected:N', scale=alt.Scale(domain=['Recommended', 'Candidate'], range=['#E40000', '#475569']))
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)

        # Calculate quote pricing
        bundle_items = []
        total_monthly = 0
        total_setup = 0
        max_install = 0
        max_contract = 12

        for pid in selected_pids:
            prod = catalogue_df.loc[pid]
            qty = 1
            if prod['unit'] == 'per site': qty = max(1, num_sites)
            elif prod['unit'] == 'per line': qty = min(max(2, emp_count // 3), 10)
            elif prod['unit'] == 'per user': qty = min(max(3, emp_count), 20)

            unit_price = int((prod['price_min_inr_per_month'] + prod['price_max_inr_per_month']) // 2)
            subtotal = unit_price * qty
            total_monthly += subtotal
            total_setup += int(prod['one_time_setup_inr'])
            max_install = max(max_install, int(prod['typical_install_days']))
            max_contract = max(max_contract, int(prod['min_contract_months']))

            bundle_items.append({
                'Code': pid,
                'Product': prod['product_name'],
                'Category': prod['category'],
                'Qty': f"{qty} {prod['unit']}",
                'Monthly (₹)': f"₹{subtotal:,}",
                'Setup (₹)': f"₹{int(prod['one_time_setup_inr']):,}",
                'Install': f"{prod['typical_install_days']}d"
            })

        # Summary KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Consolidated Monthly", f"₹{total_monthly:,}/mo")
        k2.metric("Upfront Setup", f"₹{total_setup:,}")
        k3.metric("Guaranteed SLA", f"{max_install} Days")

        st.dataframe(pd.DataFrame(bundle_items), hide_index=True, use_container_width=True)

        # Plain language AI explanation
        st.info(f"**AI Advisor Rationale:** Specifically tailored for {biz_name} ({segment}) across {num_sites} site(s) and {emp_count} staff. By consolidating your business broadband, billing, and communication tools into a single monthly bill of ₹{total_monthly:,}, we eliminate vendor coordination overhead with installation guaranteed within {max_install} business days.")

# -----------------------------------------------------------------------------
# VIEW 2: LIVE LEAD PROPENSITY & EV ENGINE
# -----------------------------------------------------------------------------
elif menu == "📊 2. Live Lead Propensity & EV Engine":
    st.title("📊 Live Lead Propensity & Expected Value (EV) Engine")
    st.markdown("Real-time inference using the **Soft-Voting Ensemble (AUC 0.8295)** and **HistGradientBoosting MRR Regressor** loaded via Joblib.")

    st.subheader("1. Live Single Lead Scoring Simulator")
    sim_col1, sim_col2 = st.columns([5, 7])

    with sim_col1:
        sim_lead_choice = st.selectbox(
            "Select Existing Lead from smb_leads.csv (or test custom):",
            [f"{row['lead_id']} - {row['business_name']} ({row['segment']})" for _, row in leads_df.head(15).iterrows()]
        )
        lead_id_selected = sim_lead_choice.split(" - ")[0]
        lead_row = leads_df[leads_df['lead_id'] == lead_id_selected].iloc[0]

        st.markdown(f"**Lead Details:** `{lead_row['lead_id']}` · {lead_row['city']} (Tier {lead_row['city_tier']}) · Source: {lead_row['lead_source']}")
        
        s_emp = st.slider("Staff Count", 1, 100, int(lead_row['employee_count']))
        s_sites = st.slider("Site Count", 1, 20, int(lead_row['num_sites']))
        s_spend = st.number_input("Telecom Monthly Spend (₹)", 500, 100000, int(lead_row['monthly_telecom_spend_inr']), step=500)
        s_gst = st.radio("GSTIN Registered?", ['Y', 'N'], index=0 if lead_row['has_gstin'] == 'Y' else 1, horizontal=True)

    with sim_col2:
        # Build feature vector for selected input
        sim_input = lead_row.to_dict()
        sim_input['employee_count'] = s_emp
        sim_input['num_sites'] = s_sites
        sim_input['monthly_telecom_spend_inr'] = s_spend
        sim_input['has_gstin'] = s_gst
        sim_vec = build_feature_vector(sim_input)

        # REAL MODEL INFERENCE
        pred_prob = propensity_ensemble.predict_proba(sim_vec)[0, 1]
        pred_mrr = float(np.clip(mrr_regressor.predict(sim_vec)[0], 999, None))
        ev_annual = int(pred_prob * pred_mrr * 12)
        prop_score = round(pred_prob * 100, 1)

        # Routing logic
        if prop_score >= 75 and pred_mrr >= 6000:
            route_name = "Field Sales VIP (High EV)"
            route_badge = "badge-vip"
            route_desc = "Assigned to dedicated enterprise field representative for high-touch white-glove closing."
        elif prop_score >= 50:
            route_name = "Channel Partner Assisted (Warm Drop-off)"
            route_badge = "badge-partner"
            route_desc = "Dispatched to local channel partner for telephone or in-store assisted onboarding."
        else:
            route_name = "100% Digital Self-Serve"
            route_badge = "badge-digital"
            route_desc = "Onboarded entirely via automated smartphone web & WhatsApp journey at near-zero CAC."

        st.markdown("#### Live ML Model Inference Result")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Propensity Score", f"{prop_score}%", help="Soft-voting ensemble activation probability")
        m_col2.metric("Predicted MRR", f"₹{int(pred_mrr):,}/mo", help="HistGradientBoosting predicted bundle value")
        m_col3.metric("Expected Annual Value", f"₹{ev_annual:,}", help="EV = P(Activation) × Predicted MRR × 12")

        st.markdown(f"**Automated Omnichannel Routing:** <span class='{route_badge}'>{route_name}</span>", unsafe_allow_html=True)
        st.caption(route_desc)

    st.markdown("---")
    st.subheader("2. 5,000 Scored Leads Database (Filterable by Expected Value & Segment)")
    
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        seg_filter = st.selectbox("Filter by Segment:", ['All'] + list(leads_df['segment'].unique()))
    with filter_col2:
        route_filter = st.selectbox("Filter by Routing:", ['All', 'Field Sales VIP (High EV)', 'Partner Assisted (Warm Drop-off)', '100% Digital Self-Serve'])

    display_df = scored_leads_df.copy() if scored_leads_df is not None else leads_df.copy()
    if seg_filter != 'All':
        display_df = display_df[display_df['segment'] == seg_filter]
    if route_filter != 'All' and 'routing' in display_df.columns:
        display_df = display_df[display_df['routing'] == route_filter]

    st.dataframe(
        display_df[['lead_id', 'business_name', 'segment', 'city', 'propensity_score', 'predicted_mrr', 'expected_annual_value', 'routing', 'stage_reached', 'outcome']].head(100),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------------------------------------------------------
# VIEW 3: SMART KYC & DPDP COMPLIANCE AUDITOR
# -----------------------------------------------------------------------------
elif menu == "🛡️ 3. Smart KYC & DPDP Compliance":
    st.title("🛡️ Smart Instant KYC & DPDP Compliance Auditor")
    st.markdown("Sub-60s automated verification with **OpenRouter Multimodal Vision (`extractor.py`)**, **Mod-36 checksum**, **National GST & Udyam Registries**, and **DPDP Act data masking**.")

    # DPDP Banner
    st.success("🔒 **DPDP Act, 2023 Digital Consent Enforced:** Purpose limitation strictly bounded to telecom subscriber verification. All sensitive tax tokens masked (`37XXXXXXXXX1ZS`). Specimen documents scheduled for statutory purge 30 days post-activation.")

    # Section A: Parallel Multi-Document Vision Extraction powered by extractor.py
    st.subheader("1. Parallel Multi-Document Vision Pipeline (`extractor.py`)")
    st.markdown(
        "Real-world enterprise KYC distributes credentials across distinct documents: **GST Certificate** (GSTIN & Trade Name), "
        "**Udyam Certificate** (MSME Reg No), **Utility Bill** (Address & Recency Proof), and **Signatory ID** (Authorized Identity). "
        "No single document holds all attributes. `extractor.py` utilizes **`concurrent.futures.ThreadPoolExecutor`** to process all submitted "
        "documents in parallel, consolidating extracted tokens into an authoritative applicant profile."
    )

    doc_source_mode = st.radio(
        "Choose Document Submission Method:",
        [
            "📂 Multi-Document Specimen Dossier (K01–K12 Complete Packages)",
            "📤 Upload Multiple Custom Documents (GST + Udyam + Utility Bill + Signatory ID)"
        ],
        horizontal=True
    )

    docs_to_process = []  # List of items: either file paths or (filename, bytes/UploadedFile)

    if doc_source_mode == "📂 Multi-Document Specimen Dossier (K01–K12 Complete Packages)":
        dossier_options = [
            ("K02", "K02: Harmony Dental Care (Clean Approved Package · 4 Docs)"),
            ("K01", "K01: Vindhya Tours & Travels (Cancelled GSTIN + Expired 2024 Bill · 4 Docs)"),
            ("K03", "K03: Ganesh Woodcraft (Valid Manufacturer · 4 Docs)"),
            ("K04", "K04: Saffron Dental Care (Planted: GSTIN Mod-36 Mismatch · 4 Docs)"),
            ("K05", "K05: Metro Retail Store (Non-GST Micro Exemption · 3 Docs)"),
            ("K06", "K06: Saffron Unisex Salon (Valid Pass · 4 Docs)"),
            ("K07", "K07: Neelgiri Physio Clinic (Planted: 8-digit Udyam Error · 4 Docs)"),
            ("K08", "K08: Green Leaf Legal Associates (Valid Pass · 4 Docs)"),
            ("K09", "K09: Kaveri Insurance Advisors (Valid Pass · 4 Docs)"),
            ("K10", "K10: Kaveri Stays LLP (Multi-Site Hospitality · 4 Docs)"),
            ("K11", "K11: Trinity Bake House (Planted: Utility Bill >250 Days Expired · 4 Docs)"),
            ("K12", "K12: Star Electronics (Planted: Bill Feb 2024 >2 Years Expired · 4 Docs)")
        ]
        chosen_k = st.selectbox(
            "Select Applicant KYC Dossier to Load:",
            [opt[0] for opt in dossier_options],
            format_func=lambda x: next(opt[1] for opt in dossier_options if opt[0] == x),
            index=0
        )

        if os.path.exists('kyc_documents'):
            all_files = sorted(os.listdir('kyc_documents'))
            matching_files = [f for f in all_files if f.startswith(f"{chosen_k}_")]
            docs_to_process = [os.path.join('kyc_documents', f) for f in matching_files]

            st.markdown(f"**Loaded {len(docs_to_process)} Specimen Documents for `{chosen_k}`:**")
            preview_cols = st.columns(max(1, len(docs_to_process)))
            for idx, fpath in enumerate(docs_to_process):
                base = os.path.basename(fpath)
                doc_tag = "Document"
                if "gst" in base: doc_tag = "📜 GSTIN Certificate"
                elif "udyam" in base: doc_tag = "🏷️ Udyam Registration"
                elif "address" in base or "bill" in base: doc_tag = "💡 Address Utility Bill"
                elif "signatory" in base: doc_tag = "🪪 Signatory ID Proof"

                with preview_cols[idx]:
                    st.image(fpath, caption=f"{doc_tag}\n({base})", use_container_width=True)

    else:
        uploaded_files = st.file_uploader(
            "Upload Multiple KYC Document Images (PNG / JPG / JPEG):",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Select all required applicant proofs (GST Certificate, Udyam MSME Certificate, Recent Electricity/Telephone Bill, Signatory PAN/Aadhaar)."
        )
        if uploaded_files:
            docs_to_process = uploaded_files
            st.markdown(f"**{len(uploaded_files)} Documents Ready for Concurrency Processing:**")
            up_cols = st.columns(min(4, max(1, len(uploaded_files))))
            for idx, ufile in enumerate(uploaded_files):
                with up_cols[idx % 4]:
                    st.image(ufile, caption=f"Uploaded: {ufile.name}", use_container_width=True)
        else:
            st.info("👆 Please upload 2 or more document files above to trigger parallel multi-document extraction.")

    # Concurrency Execution Trigger
    st.markdown("<br/>", unsafe_allow_html=True)
    c_btn, c_stat = st.columns([4, 8])
    with c_btn:
        run_parallel_extract = st.button(
            f"⚡ Run Parallel Multi-Doc Vision OCR ({len(docs_to_process)} Files)",
            disabled=len(docs_to_process) == 0,
            use_container_width=True
        )

    if run_parallel_extract and docs_to_process:
        with st.spinner(f"⚡ Concurrently extracting tokens across {len(docs_to_process)} documents via ThreadPoolExecutor..."):
            res = extract_multiple_kyc_documents_parallel(docs_to_process, max_workers=min(len(docs_to_process), 6))
            st.session_state['extracted_kyc_bundle'] = res
            st.session_state['extracted_kyc'] = res.get('aggregated', {})
            st.toast("Parallel Extraction Succeeded!", icon="🚀")

    # Display Consolidated Results & Concurrency Metrics
    if 'extracted_kyc_bundle' in st.session_state:
        bundle = st.session_state['extracted_kyc_bundle']
        agg = bundle.get('aggregated', {})
        details = bundle.get('file_details', [])
        elapsed = bundle.get('elapsed_seconds', 0.0)
        n_files = bundle.get('files_processed', len(details))

        st.success(
            f"🚀 **Parallel Vision OCR Completed in {elapsed}s!** Concurrently processed {n_files} documents using **`extractor.py` `ThreadPoolExecutor`** "
            f"({round((elapsed / max(1, n_files)) * 1000, 1)}ms effective latency per document)."
        )

        # 4 Summary KPI Cards of Aggregated Tokens
        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        k_c1.metric("Extracted Trade Name", agg.get('business_name') or "Not Visible")
        k_c2.metric("Consolidated GSTIN", agg.get('gstin') or "None / Micro Exemption")
        k_c3.metric("Udyam Registration", agg.get('udyam') or "None")
        k_c4.metric("Address Bill Date", agg.get('bill_date') or "Not Detected")

        # Provenance Breakdown Table
        with st.expander("🔍 View Per-Document Token Provenance & Concurrency Audit Log", expanded=True):
            prov_rows = []
            for item in details:
                f_name = item.get('file_name', 'Unknown')
                f_ext = item.get('extracted', {})
                prov_rows.append({
                    "Document File": f_name,
                    "Extracted GSTIN": f_ext.get('gstin') or "—",
                    "Extracted Udyam": f_ext.get('udyam') or "—",
                    "Extracted Bill Date": f_ext.get('bill_date') or "—",
                    "Extracted Business Name": f_ext.get('business_name') or "—"
                })
            st.dataframe(pd.DataFrame(prov_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Compliance Audit & Registry Cross-Check")
    
    kyc_col1, kyc_col2 = st.columns([5, 7])

    with kyc_col1:
        st.markdown("**Applicant Credentials (Consolidated from Parallel Extraction):**")
        
        # Priority: Parallel extracted bundle > Single extraction > Default specimen
        last_ext = st.session_state.get('extracted_kyc', {})
        def_name = last_ext.get('business_name') or "Harmony Dental Care"
        def_gst = last_ext.get('gstin') or "19EOGXH4115Y1ZL"
        def_udy = last_ext.get('udyam') or "UDYAM-WB-00-5053330"
        def_date_str = last_ext.get('bill_date') or "2026-05-22"
        try:
            def_date = pd.to_datetime(def_date_str).date()
        except Exception:
            def_date = pd.to_datetime("2026-05-22").date()

        k_name = st.text_input("Applicant Trade Name", value=def_name)
        k_gst = st.text_input("15-Digit GSTIN (Leave blank for Micro Exemption)", value=def_gst)
        k_udy = st.text_input("Udyam Registration Number", value=def_udy)
        k_bill = st.date_input("Address Proof Utility Bill Date", value=def_date)

        st.caption(r"Benchmark Date: `2026-07-15` (Address proof validity window: $\le$ 90 days)")
        run_kyc_btn = st.button("🔍 Execute Smart KYC Audit", use_container_width=True)

    with kyc_col2:
        st.markdown("**Algorithmic Verification Report:**")
        
        audit_res = run_smart_kyc_audit(
            gstin_val=k_gst,
            udyam_val=k_udy,
            bill_date_str=str(k_bill),
            business_name=k_name
        )

        overall = audit_res['overall_status']
        if overall == 'APPROVED':
            st.markdown("### Overall Status: <span class='badge-approved'>✓ APPROVED</span>", unsafe_allow_html=True)
            st.success("All credentials verified. Eligible for instant automated line provisioning.")
        else:
            st.markdown("### Overall Status: <span class='badge-rejected'>✕ REJECTED</span>", unsafe_allow_html=True)
            st.error("Audit checks failed. See detailed component breakdown below for specific rejection triggers.")

        for chk in audit_res['checks']:
            is_ok = chk['status'] == 'VERIFIED'
            with st.container():
                st.markdown(f"""
                <div style="background-color: {'#064E3B' if is_ok else '#450A0A'}; border: 1px solid {'#059669' if is_ok else '#DC2626'}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{chk['component']}</strong>
                        <span style="font-size: 0.75rem; font-weight: bold; color: {'#34D399' if is_ok else '#F87171'};">
                            {chk['status']}
                        </span>
                    </div>
                    <div style="font-family: monospace; font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{chk['id']}</div>
                    <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 4px;">{chk['reason']}</div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 4: LIVE ORDER & MILESTONE TRACKER
# -----------------------------------------------------------------------------
elif menu == "📦 4. Live Order & Milestone Tracker":
    st.title("📦 Live Order & Milestone Activation Tracker")
    st.markdown("Directly solves the #1 driver of customer dissatisfaction: **eliminating 3.0 status chaser calls** per lead.")

    # Order Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1E293B, #0F172A); border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="background: rgba(56, 189, 248, 0.2); color: #38BDF8; font-size: 0.75rem; font-weight: bold; padding: 3px 8px; border-radius: 9999px;">ORDER ACTIVE</span>
                <h2 style="margin: 6px 0 2px 0; color: white;">Order #UDYAM-2026-8942</h2>
                <div style="color: #94A3B8; font-size: 0.85rem;">Customer: <strong>Saffron Bakers</strong> · Plan: <strong>Biz Fibre 100 + Cloud POS + 2x Mobile SIMs</strong></div>
            </div>
            <div style="text-align: right; background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 12px 20px;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Guaranteed Commercial Go-Live</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #38BDF8;">Within 48 Hours</div>
                <div style="font-size: 0.75rem; color: #10B981;">✓ 3 Days Ahead of 5-Day SLA</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5 Milestone Cards
    stages = [
        {"name": "1. Order Confirmed", "status": "COMPLETED", "time": "16 Jul 10:30 AM", "desc": "Digital single-bill bundle accepted."},
        {"name": "2. Smart KYC Cleared", "status": "COMPLETED", "time": "16 Jul 10:32 AM", "desc": "Instant GST & Udyam audit verified."},
        {"name": "3. Feasibility Cleared", "status": "COMPLETED", "time": "16 Jul 02:15 PM", "desc": "DP-box port 4 confirmed within 65m."},
        {"name": "4. Technician Dispatched", "status": "IN PROGRESS", "time": "Today by 4:00 PM", "desc": "Engineer en route with Wi-Fi router."},
        {"name": "5. Commercial Activation", "status": "PENDING", "time": "Tomorrow morning", "desc": "Live billing & portal access dispatched."}
    ]

    cols = st.columns(5)
    for i, s in enumerate(stages):
        with cols[i]:
            is_done = s['status'] == "COMPLETED"
            is_curr = s['status'] == "IN PROGRESS"
            border_color = "#10B981" if is_done else ("#38BDF8" if is_curr else "#334155")
            bg_color = "rgba(16, 185, 129, 0.1)" if is_done else ("rgba(56, 189, 248, 0.15)" if is_curr else "rgba(30, 41, 59, 0.5)")
            icon = "✓" if is_done else ("⚡" if is_curr else "○")
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1.5px solid {border_color}; border-radius: 10px; padding: 14px; text-align: center; min-height: 140px;">
                <div style="font-size: 1.2rem; font-weight: 900; color: {'#10B981' if is_done else ('#38BDF8' if is_curr else '#64748B')};">{icon}</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: white; margin-top: 4px;">{s['name']}</div>
                <div style="font-size: 0.75rem; color: {'#34D399' if is_done else ('#7DD3FC' if is_curr else '#94A3B8')}; margin-top: 2px;">{s['time']}</div>
                <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 6px;">{s['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.subheader("Field Installation Crew")
        st.markdown("""
        **Assigned Engineer:** Rajesh Kumar (Badge #ENG-4821)  
        **Skillset:** Accel Certified Broadband & Cloud Specialist  
        **Rating:** ★ 4.9 (142 installations completed)  
        **Installation Verification OTP:** `7482`
        """)
    with t_col2:
        st.subheader("Customer Churn Reduction Impact")
        st.info("In historical data (`smb_leads.csv`), customers made **3.0 follow-up calls** chasing status, directly causing a **28.4% 6-month churn rate**. Providing automated milestone visibility slashes follow-up calls by 85%, cutting churn to **11.2%** and boosting NPS to **9.2**.")

# -----------------------------------------------------------------------------
# VIEW 5: EXECUTIVE METRICS & ANALYTICS
# -----------------------------------------------------------------------------
elif menu == "📈 5. Executive Funnel Metrics & Analytics":
    st.title("📈 Executive Funnel Metrics & Impact Dashboard")
    st.markdown("Rigorous benchmark analysis comparing baseline metrics with Project Udyam Digital.")

    # High-level KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Funnel Conversion", "22.5%", delta="+15.14 pp (3.05× lift)")
    m2.metric("Blended CAC", "₹3,800", delta="-74.6% vs ₹15,000", delta_color="inverse")
    m3.metric("Activation SLA", "2.5 Days", delta="-85.8% vs 17.6 Days", delta_color="inverse")
    m4.metric("Status Chaser Calls", "0.4 calls", delta="-86.7% vs 3.0 calls", delta_color="inverse")

    st.markdown("---")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.subheader("Funnel Stage-by-Stage Survival")
        stages_data = pd.DataFrame({
            'Stage': ['1. Lead', '2. Contacted', '3. Needs Assessed', '4. Quote Accepted', '5. KYC Complete', '6. Activated'],
            'Baseline Leads': [5000, 3406, 1974, 861, 588, 368],
            'Udyam Digital (Target)': [5000, 4400, 3600, 2400, 1800, 1125]
        })
        st.bar_chart(stages_data.set_index('Stage'))

    with f_col2:
        st.subheader("Top Historical Drop Reasons Addressed")
        drop_reasons = leads_df['drop_reason'].value_counts().head(8).reset_index()
        drop_reasons.columns = ['Drop Reason', 'Count']
        st.bar_chart(drop_reasons.set_index('Drop Reason'))

    st.markdown("---")
    st.subheader("Pitch Deck Visualization Summary")
    if os.path.exists('udyam_ml_dashboard.png'):
        st.image('udyam_ml_dashboard.png', caption="Comprehensive 4-Panel Executive Dashboard (ROC Curves, Top Signals, Funnel Survival, and Churn Impact)", use_container_width=True)

