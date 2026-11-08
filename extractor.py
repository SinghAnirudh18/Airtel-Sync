# !pip install -q -U openai

import os
import json
import base64
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "openrouter/free")

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

KYC_DIR = 'kyc_documents'

def encode_image_to_base64(image_input):
    """Reads the local PNG image or raw bytes and converts it to a base64 string for the API."""
    if isinstance(image_input, bytes):
        return base64.b64encode(image_input).decode('utf-8')
    elif isinstance(image_input, str) and os.path.exists(image_input):
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif hasattr(image_input, "getvalue"):
        return base64.b64encode(image_input.getvalue()).decode('utf-8')
    elif hasattr(image_input, "read"):
        content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        return base64.b64encode(content).decode('utf-8')
    return None

def extract_kyc_with_openrouter(image_input):
    """Passes the base64 image to OpenRouter and requests strict JSON extraction."""
    base64_image = encode_image_to_base64(image_input)
    if not base64_image:
        return {"gstin": None, "udyam": None, "bill_date": None, "business_name": None}

    prompt = """
    You are an expert KYC document auditor. Analyze this document and extract the following information strictly as a JSON object.
    If a value is not found, return null. Do NOT include markdown formatting, backticks, or any conversational text. Just the JSON.

    Expected JSON schema:
    {
      "gstin": "Extract the 15-character alphanumeric GSTIN if present, else null",
      "udyam": "Extract the Udyam number in format UDYAM-XX-00-XXXXXXX if present, else null",
      "bill_date": "Extract the utility bill date in YYYY-MM-DD format if present, else null",
      "business_name": "Extract the trade or legal business name if visible, else null"
    }
    """

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0 # Force deterministic output
        )

        # Clean the output in case the model returns markdown backticks
        raw_text = response.choices[0].message.content
        json_text = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        return {
            "gstin": data.get("gstin"),
            "udyam": data.get("udyam"),
            "bill_date": data.get("bill_date"),
            "business_name": data.get("business_name")
        }

    except Exception as e:
        print(f"Error processing document: {e}")
        return {"gstin": None, "udyam": None, "bill_date": None, "business_name": None}

def process_applicant_openrouter(applicant_id, files):
    """Aggregates the OpenRouter JSON output for all documents of a single applicant using multithreading."""
    file_paths = [os.path.join(KYC_DIR, f) if not os.path.isabs(f) and not f.startswith(KYC_DIR) else f for f in files]
    res = extract_multiple_kyc_documents_parallel(file_paths, max_workers=min(len(files), 6))
    return {
        'applicant_id': applicant_id,
        'extracted_data': res['aggregated'],
        'file_details': res['file_details'],
        'elapsed_seconds': res['elapsed_seconds']
    }

def extract_multiple_kyc_documents_parallel(image_inputs, max_workers=4):
    """
    Parallel Vision OCR Extractor across multiple uploaded documents.
    Takes a list of file paths OR tuples of (filename, bytes/stream).
    Dispatches tasks across a ThreadPoolExecutor to parse GST, Udyam, address proofs,
    and signatory IDs concurrently, consolidating all tokens into a unified applicant profile.
    """
    import time
    start_time = time.time()

    aggregated = {
        'gstin': None,
        'udyam': None,
        'bill_date': None,
        'business_name': None
    }
    file_details = []

    def _worker(item):
        fname = "document"
        payload = item
        if isinstance(item, tuple) and len(item) == 2:
            fname, payload = item
        elif isinstance(item, str):
            fname = os.path.basename(item)
        elif hasattr(item, 'name'):
            fname = item.name

        extracted = extract_kyc_with_openrouter(payload)
        return fname, extracted

    workers = max(1, min(len(image_inputs), max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(_worker, inp): inp for inp in image_inputs}

        for future in concurrent.futures.as_completed(future_to_file):
            try:
                fname, ext = future.result()
                file_details.append({
                    'file_name': fname,
                    'extracted': ext
                })
                # Consolidate tokens from whichever document holds them
                if ext.get('gstin') and not aggregated['gstin']:
                    aggregated['gstin'] = ext['gstin']
                if ext.get('udyam') and not aggregated['udyam']:
                    aggregated['udyam'] = ext['udyam']
                if ext.get('bill_date') and not aggregated['bill_date']:
                    aggregated['bill_date'] = ext['bill_date']
                if ext.get('business_name') and not aggregated['business_name']:
                    aggregated['business_name'] = ext['business_name']
            except Exception as e:
                print(f"Parallel worker failure: {e}")

    return {
        'aggregated': aggregated,
        'files_processed': len(image_inputs),
        'file_details': file_details,
        'elapsed_seconds': round(time.time() - start_time, 2)
    }

if __name__ == '__main__':
    # Execute the OpenRouter pipeline over your uploaded documents
    if os.path.exists(KYC_DIR) and len(os.listdir(KYC_DIR)) > 0:
        all_files = os.listdir(KYC_DIR)
        applicants = sorted(list(set([f[:3] for f in all_files if f.startswith('K')])))

        print(f"Starting FAST OpenRouter Vision extraction for {len(applicants)} applicants using {VISION_MODEL}...\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as app_executor:
            future_to_app = {
                app_executor.submit(process_applicant_openrouter, app_id, [f for f in all_files if f.startswith(app_id)]): app_id
                for app_id in applicants
            }

            for future in concurrent.futures.as_completed(future_to_app):
                result = future.result()
                print(f"=== Results for {result['applicant_id']} ===")
                print(json.dumps(result, indent=2))
    else:
        print(f"Waiting for images... Please ensure the PNG files are in '{KYC_DIR}'.")