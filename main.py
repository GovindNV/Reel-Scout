"""
Reel Scout — Backend API
FastAPI + yt-dlp + Gemini (Native Video & Audio Processing)
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

    # Clean the URL. Remove tracking tags (?igsh=) and strip 'www.' 
    clean_url = req.url.split("?")[0].replace("www.instagram.com", "instagram.com")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download the Media (Video or Images) and the JSON Metadata
        cmd = [
            "yt-dlp",
            # FIX: Force download of both video AND audio tracks, then merge them so Gemini can hear it
            "--format", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "--write-info-json",
            # FIX: Use autonumber to support Instagram Posts with multiple images (carousels)
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
        
        # Gather all valid media files (mp4s or jpgs)
        media_files = [f for f in files if f.startswith("media_") and not f.endswith(".json") and not f.endswith(".part") and not f.endswith(".ytdl")]
        media_files.sort()

        if not media_files:
            raise HTTPException(
                status_code=422,
                detail=f"Could not download media. It may be private or unavailable. ({result.stderr[:200]})"
            )
        
        # 2. Extract the post description from the metadata
        description = ""
        info_files = [f for f in files if f.endswith(".json")]
        if info_files:
            try:
                with open(os.path.join(tmpdir, info_files[0]), "r", encoding="utf-8") as f:
                    info_data = json.load(f)
                    description = info_data.get("description", "")
            except Exception:
                pass

        # 3. Upload all Media to Gemini API
        uploaded_media = []
        try:
            # We cap it at 5 items so massive photo carousels don't overload the API
            for mf in media_files[:5]:
                file_path = os.path.join(tmpdir, mf)
                try:
                    uv = client.files.upload(path=file_path)
                except TypeError:
                    try:
                        uv = client.files.upload(file=file_path)
                    except TypeError:
                        uv = client.files.upload(file_path)
                
                # FIX: Check if state exists. Videos need processing time, but images process instantly!
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
                            break # Status is "ACTIVE" and ready for analysis
                
                uploaded_media.append(uv)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload media to AI: {str(e)}")

        # 4. Analyze Visuals + Audio + Description simultaneously
        prompt = f"""You are a travel and food destination extractor.
I have attached media from an Instagram post (a reel video or a carousel of images).

CRITICAL INSTRUCTIONS:
1. First, WATCH the video and LISTEN closely to the audio. If it's images, look at them carefully.
2. Read any text written directly on the screen or signs in the background.
3. Then, read this post description to supplement your findings:
\"\"\"{description}\"\"\"

DO NOT just rely on the description. The audio and on-screen text are your primary sources. If someone says a place out loud or it's written on screen, it MUST be extracted.

Extract every travel destination and food spot mentioned (including restaurants, cafes, cities, beaches, landmarks, street food spots, markets, countries, and any named place).

Return ONLY a JSON object with this structure — no preamble, no markdown fences, no explanation:
{{
  "destinations": [
    {{
      "name": "Name of place",
      "type": "travel|food|both",
      "category": "e.g. Restaurant, Cafe, City, Beach, Street Food, Market, Island, Temple, Park, Hotel, Country",
      "context": "Short context on how it was mentioned (e.g., 'Spoken in audio', 'Shown on screen text', or 'Found in description')"
    }}
  ]
}}

Rules:
- type "food": restaurants, cafes, street food, food markets, specific dishes tied to a place
- type "travel": cities, countries, landmarks, beaches, parks, islands, temples, hotels
- type "both": famous food destinations that are also travel spots
- If nothing relevant is mentioned, return {{"destinations": []}}
- Do not invent or hallucinate places."""

        try:
            # Send all uploaded media items along with the prompt
            contents = uploaded_media + [prompt]
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            raw = response.text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            destinations = data.get("destinations", [])
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI returned malformed JSON. Please try again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
        finally:
            # Always clean up all the files from Gemini's server
            for uv in uploaded_media:
                try:
                    client.files.delete(name=uv.name)
                except:
                    pass

    # We repurpose the "transcript" section to show the user the extracted description
    display_transcript = description if description else "(No written description found on this post. Context extracted directly from visual/audio analysis.)"

    return AnalyzeResponse(
        url=req.url,
        transcript=display_transcript,
        destinations=[Destination(**d) for d in destinations],
    )
