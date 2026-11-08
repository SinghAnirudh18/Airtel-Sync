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

def encode_image_to_base64(image_path):
    """Reads the local PNG image and converts it to a base64 string for the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_kyc_with_openrouter(image_path):
    """Passes the base64 image to OpenRouter and requests strict JSON extraction."""
    base64_image = encode_image_to_base64(image_path)

    prompt = """
    You are an expert KYC document auditor. Analyze this document and extract the following information strictly as a JSON object.
    If a value is not found, return null. Do NOT include markdown formatting, backticks, or any conversational text. Just the JSON.

    Expected JSON schema:
    {
      "gstin": "Extract the 15-character alphanumeric GSTIN",
      "udyam": "Extract the Udyam number in format UDYAM-XX-00-XXXXXXX",
      "bill_date": "Extract the utility bill date in YYYY-MM-DD format"
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
        return json.loads(json_text)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"gstin": None, "udyam": None, "bill_date": None}

def process_applicant_openrouter(applicant_id, files):
    """Aggregates the OpenRouter JSON output for all documents of a single applicant using multithreading."""
    final_data = {'gstin': None, 'udyam': None, 'bill_date': None}

    # Process up to 4 images at the exact same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks to the thread pool
        future_to_file = {
            executor.submit(extract_kyc_with_openrouter, os.path.join(KYC_DIR, file)): file
            for file in files
        }

        # Gather results as they finish concurrently
        for future in concurrent.futures.as_completed(future_to_file):
            extracted = future.result()
            if extracted.get('gstin'): final_data['gstin'] = extracted['gstin']
            if extracted.get('udyam'): final_data['udyam'] = extracted['udyam']
            if extracted.get('bill_date'): final_data['bill_date'] = extracted['bill_date']

    return {
        'applicant_id': applicant_id,
        'extracted_data': final_data
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