import os
import re
import json
import base64
import datetime
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BENCHMARK_TODAY = pd.to_datetime(os.getenv('BENCHMARK_DATE', '2026-07-15'))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
KYC_DIR = 'kyc_documents'
MODELS_DIR = 'models'

app = FastAPI(title="Project Udyam Digital - Accel Business", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from extractor import extract_kyc_with_openrouter, process_applicant_openrouter, extract_multiple_kyc_documents_parallel

# OpenRouter Client
ai_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY
)

# -----------------------------------------------------------------------------
# Load 3 Joblib Models & Metadata from 'models/'
# -----------------------------------------------------------------------------
print("[INFO] Loading ML models from models/ directory using joblib...")
propensity_ensemble = joblib.load(os.path.join(MODELS_DIR, 'propensity_ensemble_model.pkl'))
mrr_regressor = joblib.load(os.path.join(MODELS_DIR, 'mrr_regressor_model.pkl'))
bundle_recommender = joblib.load(os.path.join(MODELS_DIR, 'bundle_recommender_model.pkl'))

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
    model_metadata = json.load(f)

FEATURE_COLUMNS = model_metadata["feature_columns"]
ACTIVE_PIDS = model_metadata["active_pids"]
print(f"[INFO] Models loaded: Propensity Ensemble (AUC: {model_metadata.get('propensity_auc')}), "
      f"MRR Regressor, and Multi-Output Bundle Recommender for {len(ACTIVE_PIDS)} products.")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
leads_df = pd.read_csv('smb_leads.csv')
catalogue_df = pd.read_csv('product_catalogue.csv').set_index('product_id')
gstin_registry = pd.read_csv('mock_gstin_registry.csv')
udyam_registry = pd.read_csv('mock_udyam_registry.csv')

# Pre-indexed dictionaries for fast lookup
gstin_dict = {row['gstin']: row.to_dict() for _, row in gstin_registry.iterrows()}
udyam_dict = {row['udyam_number']: row.to_dict() for _, row in udyam_registry.iterrows()}

# Load pre-scored leads table computed with joblib models
scored_leads_path = os.path.join(MODELS_DIR, 'scored_leads.csv')
if os.path.exists(scored_leads_path):
    scored_leads_df = pd.read_csv(scored_leads_path)
    print(f"[INFO] Loaded {len(scored_leads_df)} pre-scored leads using Joblib models.")
else:
    scored_leads_df = leads_df.copy()

# Specimen ground truth for K01-K12
SPECIMEN_GROUND_TRUTH = {
    "K01": {
        "applicant_name": "Vindhya Tours & Travels",
        "segment": "Professional offices",
        "city": "Vijayawada",
        "city_tier": 2,
        "gstin": "37IHKXV5167L1ZS",
        "udyam_number": "UDYAM-MH-00-2675962",
        "bill_date": "2024-03-24",
        "notes": "Planted errors: GSTIN status is Cancelled; Utility bill is dated 2024 (over 800 days old)."
    },
    "K02": {
        "applicant_name": "Harmony Dental Care",
        "segment": "Professional offices",
        "city": "Kolkata",
        "city_tier": 1,
        "gstin": "19EOGXH4115Y1ZL",
        "udyam_number": "UDYAM-WB-00-5053330",
        "bill_date": "2026-05-22",
        "notes": "Valid application with active GSTIN and fresh utility bill (54 days old)."
    },
    "K03": {
        "applicant_name": "Ganesh Woodcraft",
        "segment": "Small manufacturers",
        "city": "Nagpur",
        "city_tier": 2,
        "gstin": "29FGUXG3354T1ZQ",
        "udyam_number": "UDYAM-KR-00-1115424",
        "bill_date": "2026-06-15",
        "notes": "Valid manufacturer application with active registry matches (30 days old bill)."
    },
    "K04": {
        "applicant_name": "Saffron Dental Care LLP",
        "segment": "Professional offices",
        "city": "Nagpur",
        "city_tier": 2,
        "gstin": "27P25KS1936012H",
        "udyam_number": "UDYAM-MH-00-6131836",
        "bill_date": "2026-05-18",
        "notes": "Planted error: GSTIN checksum mismatch."
    },
    "K05": {
        "applicant_name": "Metro Retail Store",
        "segment": "Micro retail & services",
        "city": "Bengaluru",
        "city_tier": 1,
        "gstin": None,
        "udyam_number": None,
        "bill_date": "2026-06-11",
        "notes": "Valid Micro-retail applicant without GSTIN, verified via light-touch utility bill under DPDP."
    },
    "K06": {
        "applicant_name": "Saffron Unisex Salon",
        "segment": "Micro retail & services",
        "city": "Jaipur",
        "city_tier": 2,
        "gstin": "08YKZXS5189L1ZO",
        "udyam_number": "UDYAM-RJ-00-6577157",
        "bill_date": "2026-06-01",
        "notes": "Valid service provider with Active registrations."
    },
    "K07": {
        "applicant_name": "Neelgiri Physio Clinic",
        "segment": "Professional offices",
        "city": "Delhi",
        "city_tier": 1,
        "gstin": "29HIHXN8231E1ZI",
        "udyam_number": "UDYAM-DL-00-95655547",
        "bill_date": "2026-07-05",
        "notes": "Planted error: Udyam number has 8 digits instead of 7 (Invalid format)."
    },
    "K08": {
        "applicant_name": "Green Leaf Legal Associates",
        "segment": "Professional offices",
        "city": "Vadodara",
        "city_tier": 2,
        "gstin": "10EGQXG6446F1Z0",
        "udyam_number": "UDYAM-BR-00-7880010",
        "bill_date": "2026-05-26",
        "notes": "Valid legal office applicant with verified credentials."
    },
    "K09": {
        "applicant_name": "Kaveri Insurance Advisors",
        "segment": "Professional offices",
        "city": "Hyderabad",
        "city_tier": 1,
        "gstin": "07HBIXK4506C1ZF",
        "udyam_number": "UDYAM-TS-00-8542181",
        "bill_date": "2026-07-10",
        "notes": "Active GSTIN and fresh utility bill."
    },
    "K10": {
        "applicant_name": "Kaveri Stays LLP",
        "segment": "Small multi-site businesses",
        "city": "Mumbai",
        "city_tier": 1,
        "gstin": "24GMYXK0402E1ZQ",
        "udyam_number": "UDYAM-GJ-00-6760776",
        "bill_date": "2026-06-18",
        "notes": "Multi-site hospitality business with verified documents."
    },
    "K11": {
        "applicant_name": "Trinity Bake House Pvt Ltd",
        "segment": "Micro retail & services",
        "city": "Ludhiana",
        "city_tier": 2,
        "gstin": "33XGTXT4416F1Z3",
        "udyam_number": "UDYAM-TN-00-2649689",
        "bill_date": "2025-10-31",
        "notes": "Planted error: Utility bill dated Oct 2025 (257 days old, exceeds 90-day limit)."
    },
    "K12": {
        "applicant_name": "Star Electronics",
        "segment": "Small multi-site businesses",
        "city": "Visakhapatnam",
        "city_tier": 2,
        "gstin": "33MFTXS9954Q1Z9",
        "udyam_number": "UDYAM-TN-00-3866196",
        "bill_date": "2024-02-03",
        "notes": "Planted error: Utility bill is severely expired (Feb 2024 vs benchmark July 2026)."
    }
}

# -----------------------------------------------------------------------------
# Smart KYC Validation Helper Functions
# -----------------------------------------------------------------------------
def validate_gstin_checksum(gstin_str: str):
    """Validates 15-char GSTIN structure (accepting 'X' in 4th PAN char) + Mod-36 checksum."""
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

def run_kyc_audit(gstin_val=None, udyam_val=None, bill_date_str=None, business_name=None):
    report = {
        'dpdp_consent_verified': True,
        'overall_status': 'APPROVED',
        'overall_badge': 'success',
        'checks': [],
        'timestamp': datetime.datetime.now().isoformat()
    }

    # 1. GSTIN Check
    if gstin_val and str(gstin_val).strip() != "" and str(gstin_val).lower() != "none":
        gstin_clean = str(gstin_val).strip().upper()
        masked_gst = gstin_clean[:2] + "XXXXXXXXX" + gstin_clean[-4:] if len(gstin_clean) >= 6 else gstin_clean
        ok_fmt, msg_fmt = validate_gstin_checksum(gstin_clean)
        if not ok_fmt:
            report['checks'].append({
                'component': 'GSTIN Format & Checksum',
                'identifier': masked_gst,
                'status': 'REJECTED',
                'badge': 'error',
                'reason': msg_fmt
            })
            report['overall_status'] = 'REJECTED'
            report['overall_badge'] = 'error'
        else:
            match = gstin_registry[gstin_registry['gstin'] == gstin_clean]
            if match.empty:
                report['checks'].append({
                    'component': 'GSTIN Registry Lookup',
                    'identifier': masked_gst,
                    'status': 'REJECTED',
                    'badge': 'error',
                    'reason': 'GSTIN not found in Mock National GST Registry'
                })
                report['overall_status'] = 'REJECTED'
                report['overall_badge'] = 'error'
            else:
                row = match.iloc[0]
                if row['status'] != 'Active':
                    report['checks'].append({
                        'component': 'GSTIN Status Check',
                        'identifier': masked_gst,
                        'status': 'REJECTED',
                        'badge': 'error',
                        'reason': f"GSTIN status is '{row['status']}'. Only 'Active' status is eligible."
                    })
                    report['overall_status'] = 'REJECTED'
                    report['overall_badge'] = 'error'
                else:
                    report['checks'].append({
                        'component': 'GSTIN Verification',
                        'identifier': masked_gst,
                        'status': 'VERIFIED',
                        'badge': 'success',
                        'reason': f"Active GSTIN verified for '{row['trade_name']}' ({row['constitution']})"
                    })

    # 2. Udyam Check
    if udyam_val and str(udyam_val).strip() != "" and str(udyam_val).lower() != "none":
        udyam_clean = str(udyam_val).strip().upper()
        masked_udy = udyam_clean[:9] + "XXXX" + udyam_clean[-3:] if len(udyam_clean) >= 12 else udyam_clean
        if not re.match(r'^UDYAM-[A-Z]{2}-00-[0-9]{7}$', udyam_clean):
            report['checks'].append({
                'component': 'Udyam Format Check',
                'identifier': masked_udy,
                'status': 'REJECTED',
                'badge': 'error',
                'reason': 'Invalid Udyam structure (Must follow UDYAM-XX-00-XXXXXXX with 7 numeric digits)'
            })
            report['overall_status'] = 'REJECTED'
            report['overall_badge'] = 'error'
        else:
            match_u = udyam_registry[udyam_registry['udyam_number'] == udyam_clean]
            if match_u.empty:
                report['checks'].append({
                    'component': 'Udyam Registry Lookup',
                    'identifier': masked_udy,
                    'status': 'REJECTED',
                    'badge': 'error',
                    'reason': 'Udyam number not registered in MSME Portal Database'
                })
                report['overall_status'] = 'REJECTED'
                report['overall_badge'] = 'error'
            else:
                u_row = match_u.iloc[0]
                report['checks'].append({
                    'component': 'Udyam Verification',
                    'identifier': masked_udy,
                    'status': 'VERIFIED',
                    'badge': 'success',
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
                    'identifier': f"Bill Date: {bill_date_str}",
                    'status': 'REJECTED',
                    'badge': 'error',
                    'reason': f"Utility bill is {age_days} days old relative to benchmark 2026-07-15 (Exceeds maximum 90-day validity window)"
                })
                report['overall_status'] = 'REJECTED'
                report['overall_badge'] = 'error'
            elif age_days < 0:
                report['checks'].append({
                    'component': 'Address Proof Recency',
                    'identifier': f"Bill Date: {bill_date_str}",
                    'status': 'REJECTED',
                    'badge': 'error',
                    'reason': f"Utility bill has a future date relative to benchmark 2026-07-15"
                })
                report['overall_status'] = 'REJECTED'
                report['overall_badge'] = 'error'
            else:
                report['checks'].append({
                    'component': 'Address Proof Recency',
                    'identifier': f"Bill Date: {bill_date_str}",
                    'status': 'VERIFIED',
                    'badge': 'success',
                    'reason': f"Utility bill is recent ({age_days} days old, well within 90-day SLA window)"
                })
        except Exception as e:
            report['checks'].append({
                'component': 'Address Proof Recency',
                'identifier': f"Bill Date: {bill_date_str}",
                'status': 'FLAGGED',
                'badge': 'warning',
                'reason': f"Date parsing error: {str(e)}"
            })

    if not report['checks']:
        report['checks'].append({
            'component': 'Micro Non-GST Verification',
            'identifier': 'Tier-2 Micro Exception',
            'status': 'VERIFIED',
            'badge': 'success',
            'reason': 'Micro-retail tier verified via digital self-declaration under DPDP Act purpose limitation.'
        })

    return report

# -----------------------------------------------------------------------------
# Recommendation & Catalogue Engine
# -----------------------------------------------------------------------------
class AdvisorRequest(BaseModel):
    business_name: str
    business_type: str
    segment: str
    city_tier: int
    employee_count: int
    num_sites: int
    has_gstin: str
    current_spend_inr: float
    preferred_language: Optional[str] = "English"
    primary_need: Optional[str] = "Reliable Internet & UPI Payments"

@app.post("/api/advisor/recommend")
def get_recommendation(req: AdvisorRequest):
    tier = int(req.city_tier)
    emp = int(req.employee_count)
    sites = int(req.num_sites)
    gst_ok = req.has_gstin.upper() == 'Y'

    # 1. Filter eligible products
    eligible_pids = []
    ineligible_reasons = {}

    for pid, prod in catalogue_df.iterrows():
        allowed_tiers = [int(t) for t in str(prod['available_city_tiers']).split(';')]
        if tier not in allowed_tiers:
            ineligible_reasons[pid] = f"Not available in Tier {tier} cities"
            continue
        if emp < int(prod['min_employees']):
            ineligible_reasons[pid] = f"Requires min {prod['min_employees']} employees"
            continue
        if sites < int(prod['min_sites']):
            ineligible_reasons[pid] = f"Requires min {prod['min_sites']} sites"
            continue
        if prod['requires_gstin'] == 'Y' and not gst_ok:
            ineligible_reasons[pid] = "Requires an active GSTIN"
            continue
        eligible_pids.append(pid)

    # 2. Select base connectivity + complementary products based on segment
    selected_pids = []
    if req.segment == 'Micro retail & services':
        # Prefer P04 (5G AirFiber 100) or P01 (Biz Fibre 100) + P19 (POS) + P15 (WhatsApp)
        if 'P04' in eligible_pids:
            selected_pids.append('P04')
        elif 'P01' in eligible_pids:
            selected_pids.append('P01')
        if 'P19' in eligible_pids: selected_pids.append('P19')
        if 'P15' in eligible_pids and gst_ok: selected_pids.append('P15')
        if 'P07' in eligible_pids: selected_pids.append('P07')

    elif req.segment == 'Professional offices':
        # Prefer P02 (Biz Fibre 300) + P16 (Email/Office) + P20 (Security) + P11 (Static IP)
        if 'P02' in eligible_pids:
            selected_pids.append('P02')
        elif 'P01' in eligible_pids:
            selected_pids.append('P01')
        if 'P16' in eligible_pids: selected_pids.append('P16')
        if 'P20' in eligible_pids: selected_pids.append('P20')
        if 'P11' in eligible_pids: selected_pids.append('P11')

    elif req.segment == 'Small multi-site businesses':
        # Multi-site: P02 or P05 (ILL 20) + P09 (Failover) + P23 (Cloud CCTV) + P10 (SD-WAN if eligible)
        if 'P02' in eligible_pids:
            selected_pids.append('P02')
        elif 'P05' in eligible_pids:
            selected_pids.append('P05')
        if 'P09' in eligible_pids: selected_pids.append('P09')
        if 'P23' in eligible_pids: selected_pids.append('P23')
        if 'P10' in eligible_pids: selected_pids.append('P10')

    else: # Small manufacturers
        if 'P05' in eligible_pids:
            selected_pids.append('P05')
        elif 'P03' in eligible_pids:
            selected_pids.append('P03')
        elif 'P06' in eligible_pids:
            selected_pids.append('P06')
        if 'P21' in eligible_pids: selected_pids.append('P21')
        if 'P24' in eligible_pids: selected_pids.append('P24')
        if 'P25' in eligible_pids: selected_pids.append('P25')

    # Dependency check (requires_one_of)
    for pid in list(selected_pids):
        req_dep = str(catalogue_df.loc[pid, 'requires_one_of'])
        if req_dep != 'nan' and req_dep != '':
            options = req_dep.split('|')
            if not any(o in selected_pids for o in options):
                valid_opts = [o for o in options if o in eligible_pids]
                if valid_opts:
                    selected_pids.insert(0, valid_opts[0])
                else:
                    selected_pids.remove(pid)

    # 3. Calculate pricing & build bundle items
    bundle_items = []
    total_monthly = 0
    total_setup = 0
    max_install = 0
    max_contract = 12

    for pid in selected_pids:
        prod = catalogue_df.loc[pid]
        qty = 1
        if prod['unit'] == 'per site':
            qty = max(1, sites)
        elif prod['unit'] == 'per line':
            qty = min(max(2, emp // 3), 10)
        elif prod['unit'] == 'per user':
            qty = min(max(3, emp), 20)
        elif prod['unit'] == 'per camera':
            qty = min(max(2, sites * 2), 8)

        unit_price = int((prod['price_min_inr_per_month'] + prod['price_max_inr_per_month']) // 2)
        line_price = unit_price * qty
        total_monthly += line_price
        total_setup += int(prod['one_time_setup_inr'])
        max_install = max(max_install, int(prod['typical_install_days']))
        max_contract = max(max_contract, int(prod['min_contract_months']))

        bundle_items.append({
            'product_id': pid,
            'name': prod['product_name'],
            'category': prod['category'],
            'description': prod['description'],
            'quantity': qty,
            'unit': prod['unit'],
            'unit_price': unit_price,
            'monthly_subtotal': line_price,
            'setup_fee': int(prod['one_time_setup_inr']),
            'install_days': int(prod['typical_install_days'])
        })

    # AI Plain Language Narrative
    lang = req.preferred_language or "English"
    if lang.lower() == 'hindi':
        pitch = f"आपके {req.business_type} व्यवसाय के लिए हमने एक सिंगल-बिल डिजिटल बंडल तैयार किया है। इसमें {sites} साइट्स के लिए हाई-स्पीड इंटरनेट, क्लाउड टूल्स और 24x7 मॉनिटरिंग शामिल है। आपको अलग-अलग वेंडर्स के पीछे भागने की आवश्यकता नहीं है, सब कुछ ₹{total_monthly:,}/माह में एक बिल पर सक्रिय हो जाएगा।"
    else:
        pitch = f"Custom-designed for your {req.business_type} with {sites} location(s) and {emp} employees. This single-bill bundle eliminates multiple vendor invoices by combining guaranteed high-speed connectivity with essential business security and customer engagement tools, activated in just {max_install} business days."

    return {
        'business_name': req.business_name,
        'segment': req.segment,
        'preferred_language': lang,
        'explanation': pitch,
        'items': bundle_items,
        'transparent_monthly_total': total_monthly,
        'one_time_setup_total': total_setup,
        'contract_months': max_contract,
        'estimated_activation_days': max_install,
        'eligible_products_count': len(eligible_pids),
        'ineligible_reasons': ineligible_reasons
    }

# -----------------------------------------------------------------------------
# Dynamic Quote Re-calculator
# -----------------------------------------------------------------------------
class QuoteItem(BaseModel):
    product_id: str
    quantity: int

class QuoteCalculateRequest(BaseModel):
    items: List[QuoteItem]
    city_tier: int = 1
    employee_count: int = 5
    num_sites: int = 1
    has_gstin: str = 'Y'

@app.post("/api/quote/calculate")
def recalculate_quote(req: QuoteCalculateRequest):
    total_monthly = 0
    total_setup = 0
    max_install = 0
    max_contract = 1
    itemized = []
    selected_pids = [item.product_id for item in req.items if item.product_id in catalogue_df.index]

    # Validate dependencies
    dependency_warnings = []
    for pid in selected_pids:
        req_dep = str(catalogue_df.loc[pid, 'requires_one_of'])
        if req_dep != 'nan' and req_dep != '':
            options = req_dep.split('|')
            if not any(o in selected_pids for o in options):
                names = [catalogue_df.loc[o, 'product_name'] for o in options if o in catalogue_df.index]
                dependency_warnings.append(f"{catalogue_df.loc[pid, 'product_name']} requires at least one of: {', '.join(names)}")

    for item in req.items:
        if item.product_id not in catalogue_df.index:
            continue
        prod = catalogue_df.loc[item.product_id]
        unit_price = int((prod['price_min_inr_per_month'] + prod['price_max_inr_per_month']) // 2)
        qty = max(1, item.quantity)
        subtotal = unit_price * qty
        total_monthly += subtotal
        total_setup += int(prod['one_time_setup_inr'])
        max_install = max(max_install, int(prod['typical_install_days']))
        max_contract = max(max_contract, int(prod['min_contract_months']))

        itemized.append({
            'product_id': item.product_id,
            'name': prod['product_name'],
            'category': prod['category'],
            'quantity': qty,
            'unit': prod['unit'],
            'unit_price': unit_price,
            'monthly_subtotal': subtotal,
            'setup_fee': int(prod['one_time_setup_inr']),
            'install_days': int(prod['typical_install_days'])
        })

    return {
        'items': itemized,
        'total_monthly_inr': total_monthly,
        'total_setup_fee_inr': total_setup,
        'estimated_install_days': max_install,
        'min_contract_months': max_contract,
        'dependency_warnings': dependency_warnings
    }

# -----------------------------------------------------------------------------
# Smart KYC Verification Endpoints
# -----------------------------------------------------------------------------
class SpecimenKYCRequest(BaseModel):
    applicant_id: str

@app.post("/api/kyc/verify-specimen")
def verify_specimen(req: SpecimenKYCRequest):
    app_id = req.applicant_id.upper()
    if app_id not in SPECIMEN_GROUND_TRUTH:
        raise HTTPException(status_code=404, detail="Specimen applicant not found")

    data = SPECIMEN_GROUND_TRUTH[app_id]
    audit = run_kyc_audit(
        gstin_val=data['gstin'],
        udyam_val=data['udyam_number'],
        bill_date_str=data['bill_date'],
        business_name=data['applicant_name']
    )
    audit['applicant_id'] = app_id
    audit['applicant_name'] = data['applicant_name']
    audit['segment'] = data['segment']
    audit['city'] = data['city']
    audit['notes'] = data['notes']
    return audit

@app.post("/api/kyc/upload-and-verify")
async def upload_and_verify(
    gstin: Optional[str] = Form(None),
    udyam: Optional[str] = Form(None),
    bill_date: Optional[str] = Form(None),
    business_name: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None)
):
    extracted_gstin = gstin
    extracted_udyam = udyam
    extracted_bill_date = bill_date

    # Consolidate multiple or single uploaded files
    all_uploads: List[UploadFile] = []
    if files:
        all_uploads.extend(files)
    if file:
        all_uploads.append(file)

    extraction_stats = {}
    if all_uploads:
        try:
            # Read files into (filename, bytes) items for concurrent worker pool
            doc_items = []
            for up in all_uploads:
                contents = await up.read()
                doc_items.append((up.filename, contents))

            res = extract_multiple_kyc_documents_parallel(doc_items, max_workers=min(len(doc_items), 6))
            extracted_data = res.get('aggregated', {})
            extraction_stats = {
                'files_processed': res.get('files_processed', len(doc_items)),
                'elapsed_seconds': res.get('elapsed_seconds', 0.0),
                'file_details': res.get('file_details', [])
            }

            if extracted_data.get('gstin') and not extracted_gstin:
                extracted_gstin = extracted_data['gstin']
            if extracted_data.get('udyam') and not extracted_udyam:
                extracted_udyam = extracted_data['udyam']
            if extracted_data.get('bill_date') and not extracted_bill_date:
                extracted_bill_date = extracted_data['bill_date']
            if extracted_data.get('business_name') and not business_name:
                business_name = extracted_data['business_name']
        except Exception as e:
            print(f"Extractor parallel error: {e}")

    audit = run_kyc_audit(
        gstin_val=extracted_gstin,
        udyam_val=extracted_udyam,
        bill_date_str=extracted_bill_date,
        business_name=business_name
    )
    audit['extracted_raw'] = {
        'gstin': extracted_gstin,
        'udyam': extracted_udyam,
        'bill_date': extracted_bill_date,
        'business_name': business_name
    }
    audit['extraction_stats'] = extraction_stats
    return audit

# -----------------------------------------------------------------------------
# Live Order Tracker Endpoint
# -----------------------------------------------------------------------------
@app.get("/api/tracker/{order_id}")
def get_order_tracking(order_id: str):
    # Deterministic simulation of tracker stages based on order_id
    stages = [
        {"id": 1, "name": "Digital Order Placed", "status": "completed", "date": "2026-07-16 10:30 AM", "desc": "Custom single-bill bundle confirmed."},
        {"id": 2, "name": "Smart KYC Verified", "status": "completed", "date": "2026-07-16 10:32 AM", "desc": "Instant GST & Udyam check passed under DPDP consent."},
        {"id": 3, "name": "Fibre Feasibility Cleared", "status": "completed", "date": "2026-07-16 02:15 PM", "desc": "DP-box port 4 available within 65 metres."},
        {"id": 4, "name": "Field Technician Dispatched", "status": "in_progress", "date": "Expected Today by 4:00 PM", "desc": "Engineer assigned with Wi-Fi router & ONT kit."},
        {"id": 5, "name": "Commercial Activation", "status": "pending", "date": "Guaranteed within 48 Hrs", "desc": "Single invoice generated & portal login dispatched via SMS."}
    ]

    return {
        "order_id": order_id,
        "customer_name": "Saffron Bakers",
        "bundle": "Biz Fibre 300 + Cloud CCTV + 2x Postpaid SIMs",
        "monthly_charge_inr": 3198,
        "guaranteed_activation_date": "2026-07-18",
        "days_remaining": 2,
        "stages": stages,
        "technician": {
            "name": "Rajesh Kumar (Accel Certified Engineer)",
            "phone": "+91 98450 XXXXX",
            "rating": 4.9,
            "badge": "Fibre & Cloud Specialist",
            "installation_otp": "7482"
        },
        "customer_chase_reduction_metric": "Live milestone visibility eliminates up to 3 chaser calls per lead, reducing 6-month churn by 42%."
    }

# -----------------------------------------------------------------------------
# Back-Office ML Lead Router & Funnel Analytics
# -----------------------------------------------------------------------------
@app.get("/api/backoffice/leads")
def get_backoffice_leads(limit: int = 50, segment: Optional[str] = None):
    sub = scored_leads_df.copy()
    if segment and segment != 'All':
        sub = sub[sub['segment'] == segment]

    leads_list = []
    for _, row in sub.head(limit).iterrows():
        leads_list.append({
            'lead_id': row['lead_id'],
            'business_name': row['business_name'],
            'segment': row['segment'],
            'city': row['city'],
            'city_tier': int(row['city_tier']),
            'lead_source': row['lead_source'],
            'employee_count': int(row['employee_count']),
            'current_spend': int(row['monthly_telecom_spend_inr']),
            'stage_reached': row['stage_reached'],
            'outcome': row['outcome'],
            'drop_reason': str(row['drop_reason']) if pd.notna(row['drop_reason']) else 'N/A',
            'propensity_score': float(row['propensity_score']),
            'predicted_mrr': int(row['predicted_mrr']),
            'expected_annual_value': int(row['expected_annual_value']),
            'routing': str(row['routing']),
            'route_badge': str(row['route_badge'])
        })

    return {
        'total_leads_sampled': len(leads_list),
        'leads': sorted(leads_list, key=lambda x: x['expected_annual_value'], reverse=True)
    }

@app.get("/api/backoffice/funnel-metrics")
def get_funnel_metrics():
    stages_counts = leads_df['stage_reached'].value_counts()
    drop_reasons = leads_df['drop_reason'].value_counts().head(8).to_dict()

    return {
        "funnel": {
            "Leads Generated": 5000,
            "Contacted": 3406,
            "Needs Assessed": 1974,
            "Quote Accepted": 861,
            "KYC Complete": 588,
            "Activated": 368
        },
        "key_metrics": {
            "current_conversion_rate": "7.36%",
            "target_digital_conversion_rate": "22.50%",
            "current_activation_sla_days": 17.6,
            "target_digital_sla_days": 2.5,
            "current_cac_inr": 15000,
            "target_digital_cac_inr": 3800,
            "avg_chaser_calls": 3.0,
            "target_chaser_calls": 0.4
        },
        "top_drop_reasons": drop_reasons
    }

# -----------------------------------------------------------------------------
# Mount Static Frontend
# -----------------------------------------------------------------------------
os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == '__main__':
    import uvicorn
    print("[INFO] Starting Project Udyam Digital Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
