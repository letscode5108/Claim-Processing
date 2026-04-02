
import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from graph import claim_graph

app = FastAPI(
    title="Claim Processing Pipeline",
    description="Processes PDF claims using LangGraph + Gemini",
    version="1.0.0"
)


@app.get("/")
def health_check():
    return {"status": "running", "message": "Claim Processing Pipeline is live"}

@app.post("/api/debug")
async def debug_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    from utils.pdf_utils import extract_pages_as_text
    texts = extract_pages_as_text(pdf_bytes)
    return {
        page: {
            "char_count": len(text),
            "preview": text[:200]
        }
        for page, text in texts.items()
    }

@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(...),
    file: UploadFile = File(...)
):
   
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    try:
        initial_state = {
            "pdf_bytes":          pdf_bytes,
            "claim_id":           claim_id,
            "page_classification": {},
            "id_data":            {},
            "discharge_data":     {},
            "bill_data":          {},
            "final_result":       {}
        }

        final_state = claim_graph.invoke(initial_state)
        return JSONResponse(content=final_state["final_result"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)