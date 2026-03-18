"""
Reel Scout — Backend API
FastAPI + yt-dlp + Gemini (Two-Pass Native Processing)
"""

import os
import json
import time
import tempfile
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /frontend folder
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

class AnalyzeRequest(BaseModel):
    url: str

class Destination(BaseModel):
    name: str
    type: str
    category: str
    context: str

class AnalyzeResponse(BaseModel):
    url: str
    transcript: str
    destinations: list[Destination]

@app.get("/")
def root():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"status": "Reel Scout API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if "instagram.com" not in req.url:
        raise HTTPException(status_code=400, detail="URL must be an Instagram link.")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Clean the URL
    clean_url = req.url.split("?")[0].replace("www.instagram.com", "instagram.com")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download the Media
        cmd = [
            "yt-dlp",
            "--merge-output-format", "mp4",
            "--write-info-json",
            "--output", f"{tmpdir}/media_%(autonumber)s.%(ext)s",
            "--user-agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "--force-ipv4",
            "--extractor-retries", "3",
            "--quiet",
            "--no-warnings",
            clean_url,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        files = os.listdir(tmpdir)
        media_files = [f for f in files if f.startswith("media_") and not f.endswith(".json") and not f.endswith(".part") and not f.endswith(".ytdl")]
        media_files.sort()

        if not media_files:
            raise HTTPException(
                status_code=422,
                detail=f"Could not download media. It may be private, a protected photo carousel, or unavailable. ({result.stderr[:200]})"
            )
        
        # 2. Extract description
        description = ""
        info_files = [f for f in files if f.endswith(".json")]
        if info_files:
            try:
                with open(os.path.join(tmpdir, info_files[0]), "r", encoding="utf-8") as f:
                    info_data = json.load(f)
                    description = info_data.get("description", "")
            except Exception:
                pass

        # 3. Upload Media
        uploaded_media = []
        try:
            for mf in media_files[:5]:
                file_path = os.path.join(tmpdir, mf)
                try:
                    uv = client.files.upload(path=file_path)
                except TypeError:
                    try:
                        uv = client.files.upload(file=file_path)
                    except TypeError:
                        uv = client.files.upload(file_path)
                
                if hasattr(uv, "state") and uv.state is not None:
                    while True:
                        current_state = uv.state
                        state_str = current_state.name if hasattr(current_state, "name") else str(current_state)
                        state_str = state_str.upper()

                        if "PROCESSING" in state_str:
                            time.sleep(2)
                            uv = client.files.get(name=uv.name)
                        elif "FAILED" in state_str:
                            raise Exception("Media processing failed inside the Gemini API.")
                        else:
                            break
                uploaded_media.append(uv)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload media to AI: {str(e)}")

        # ==========================================
        # PASS 1: The Natural Analysis (Your Idea!)
        # ==========================================
        prompt_pass_1 = f"""I have directly uploaded media files (a video reel or images) from an Instagram post.
        
        Please WATCH the media closely, LISTEN to the audio, and read this post description:
        \"\"\"{description}\"\"\"

        I want you to comment and list out all the locations mentioned in this video and description. 
        Explain exactly where they are mentioned (e.g., spoken, written on screen, or in the caption). 
        Be thorough and conversational."""

        try:
            contents = uploaded_media + [prompt_pass_1]
            response_pass_1 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            ai_commentary = response_pass_1.text.strip()
            
        except Exception as e:
            # Clean up files before raising error
            for uv in uploaded_media:
                try: client.files.delete(name=uv.name)
                except: pass
            raise HTTPException(status_code=500, detail=f"AI analysis (Pass 1) failed: {str(e)}")

        # Clean up the heavy media files immediately after Pass 1 is done!
        for uv in uploaded_media:
            try: client.files.delete(name=uv.name)
            except: pass

        # ==========================================
        # PASS 2: JSON Formatting Extraction
        # ==========================================
        prompt_pass_2 = f"""You are a data extractor. Convert the following AI commentary into a strict JSON object.

        Commentary to convert:
        \"\"\"{ai_commentary}\"\"\"

        Return ONLY a JSON object with this EXACT structure:
        {{
          "destinations": [
            {{
              "name": "Name of place",
              "type": "travel|food|both",
              "category": "e.g. Restaurant, Cafe, City, Beach, Street Food, Market, Island, Temple, Park, Hotel, Country",
              "context": "Short context on how it was mentioned based on the commentary"
            }}
          ]
        }}
        
        If no relevant places are mentioned in the commentary, return {{"destinations": []}}."""

        try:
            response_pass_2 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt_pass_2],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            raw = response_pass_2.text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            destinations = data.get("destinations", [])
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI returned malformed JSON during extraction. Please try again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI extraction (Pass 2) failed: {str(e)}")

    # We show the user the pure conversational text from Pass 1 when they click "Show AI Analysis"
    display_transcript = f"AI COMMENTARY:\n{ai_commentary}"

    return AnalyzeResponse(
        url=req.url,
        transcript=display_transcript,
        destinations=[Destination(**d) for d in destinations],
    )
