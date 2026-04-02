

import json
import os
from dotenv import load_dotenv
from groq import Groq
from utils.pdf_utils import extract_pages_as_images, extract_selected_pages_bytes

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT = """You are a medical discharge summary extractor for insurance claims.
Extract from this document image:
- hospital_name
- admission_date
- discharge_date
- total_days_admitted
- primary_diagnosis
- secondary_diagnosis (list or null)
- treating_physician
- physician_registration_number
- department
- procedures_performed (list or null)
- discharge_condition

Respond ONLY with valid JSON. No explanation. No markdown. If not found use null.
"""



# Replace run_discharge_agent:
def run_discharge_agent(pdf_bytes: bytes, page_nums: list) -> dict:
    if not page_nums:
        return {"error": "No discharge summary pages assigned"}

    selected_bytes = extract_selected_pages_bytes(pdf_bytes, page_nums)
    page_images = extract_pages_as_images(selected_bytes)

    content = [{"type": "text", "text": PROMPT}]
    for img_b64 in page_images.values():
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "error": "Failed to parse JSON"}