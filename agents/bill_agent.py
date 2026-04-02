

import json
import os
from dotenv import load_dotenv
from groq import Groq
from utils.pdf_utils import extract_pages_as_images, extract_selected_pages_bytes

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT = """You are a medical bill extractor for insurance claims.
Extract ALL line items from this itemized bill image and calculate totals.

Return:
- items: list of {description, quantity, unit_price, total_price}
- subtotal
- taxes
- grand_total
- currency

Respond ONLY with valid JSON. No explanation. No markdown. If not found use null.
"""



# Replace run_bill_agent:
def run_bill_agent(pdf_bytes: bytes, page_nums: list) -> dict:
    if not page_nums:
        return {"error": "No itemized bill pages assigned"}

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