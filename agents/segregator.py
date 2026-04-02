import json
import os
import time
from typing import Dict, List
from dotenv import load_dotenv
from groq import Groq
from utils.pdf_utils import extract_pages_as_text, extract_pages_as_images

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other"
]

SYSTEM_PROMPT = """You are a medical document classifier for insurance claim processing.
You will be given the text content of a PDF page.
Classify it into EXACTLY one of these types:
- claim_forms
- cheque_or_bank_details
- identity_document
- itemized_bill
- discharge_summary
- prescription
- investigation_report
- cash_receipt
- other

Respond ONLY with valid JSON. No explanation. No markdown.
Format: {"page": <page_number>, "doc_type": "<type>", "confidence": <0.0-1.0>}
"""



def classify_page(page_num: int, page_text: str, page_image_b64: str) -> Dict:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": f"Page number: {page_num}\n{SYSTEM_PROMPT}"
                    }
                ]
            }
        ],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        if result.get("doc_type") not in DOCUMENT_TYPES:
            result["doc_type"] = "other"
        return result
    except json.JSONDecodeError:
        return {"page": page_num, "doc_type": "other", "confidence": 0.0}

def run_segregator(pdf_bytes: bytes) -> Dict[str, List[int]]:
    """
    Classify all pages in the PDF.
    Returns: { "doc_type": [page_numbers], ... }
    e.g. { "identity_document": [0, 2], "itemized_bill": [3, 4] }
    """
    page_texts = extract_pages_as_text(pdf_bytes)
    page_images = extract_pages_as_images(pdf_bytes)

    classification: Dict[str, List[int]] = {doc_type: [] for doc_type in DOCUMENT_TYPES}

    for page_num in page_texts:
        result = classify_page(
            page_num=page_num,
            page_text=page_texts[page_num],
            page_image_b64=page_images[page_num]
        )
        doc_type = result.get("doc_type", "other")
        classification[doc_type].append(page_num)
        print(f"  Page {page_num} → {doc_type} (confidence: {result.get('confidence', '?')})")
        time.sleep(6)  # To avoid hitting rate limits

    # Remove empty categories
    return {k: v for k, v in classification.items() if v}

