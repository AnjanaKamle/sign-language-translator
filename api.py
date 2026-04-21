import sys
import os

# Ensure the local source is importable without pip install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sign_language_translator as slt

app = FastAPI(title="ISL Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the translation model
try:
    model = slt.models.ConcatenativeSynthesis(
        text_language="english", 
        sign_language="in-sl", 
        sign_format="video" 
    )
except Exception as e:
    print(f"Failed to load model. Error: {e}")
    model = None

class TranslationRequest(BaseModel):
    text: str

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Translation model is not initialized.")
    
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        sign = model.translate(text)
        os.makedirs("temp", exist_ok=True)
        out_path = os.path.join("temp", "translation_output.mp4")
        sign.save(out_path, overwrite=True, codec="mp4v")
        return FileResponse(out_path, media_type="video/mp4", filename="translation.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n✨ ISL Translator running at: http://127.0.0.1:8000\n")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
