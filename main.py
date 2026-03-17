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

    # MAGIC FIX 1: Clean the URL. Remove tracking tags (?igsh=) and strip 'www.' 
    clean_url = req.url.split("?")[0].replace("www.instagram.com", "instagram.com")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download the MP4 Video and the JSON Metadata (for the description)
        cmd = [
            "yt-dlp",
            "--format", "best",
            "--write-info-json",
            "--output", f"{tmpdir}/video.%(ext)s",
            "--user-agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "--force-ipv4",
            "--extractor-retries", "3",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            clean_url,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        video_path = os.path.join(tmpdir, "video.mp4")
        info_path = os.path.join(tmpdir, "video.info.json")

        # Fallback in case yt-dlp saves it under a slightly different filename
        if not os.path.exists(video_path):
            files = os.listdir(tmpdir)
            video_files = [f for f in files if f.startswith("video.") and not f.endswith(".json")]
            if video_files:
                video_path = os.path.join(tmpdir, video_files[0])
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not download reel. It may be private or unavailable. ({result.stderr[:200]})"
                )
        
        # 2. Extract the post description from the metadata
        description = ""
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_data = json.load(f)
                    description = info_data.get("description", "")
            except Exception:
                pass

        # 3. Upload the Video to Gemini API
        try:
            # MAGIC FIX 2: Google frequently renames this parameter.
            # This nested try-except block makes the app 100% version-proof!
            try:
                uploaded_video = client.files.upload(path=video_path)
            except TypeError:
                try:
                    uploaded_video = client.files.upload(file=video_path)
                except TypeError:
                    uploaded_video = client.files.upload(video_path)
            
            # Wait for Gemini to process the video file
            while uploaded_video.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_video = client.files.get(name=uploaded_video.name)
                
            if uploaded_video.state.name == "FAILED":
                raise Exception("Video processing failed inside the Gemini API.")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload video to AI: {str(e)}")

        # 4. Analyze Video + Audio + Description simultaneously
        prompt = f"""You are a travel and food destination extractor.
Watch the provided Instagram reel, listen to the audio, and carefully read any text written on the screen.
Also, factor in this description from the post:
\"\"\"{description}\"\"\"

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
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_video, prompt],
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
            # Always clean up the video from Gemini's server so you don't hit your storage limits!
            try:
                client.files.delete(name=uploaded_video.name)
            except:
                pass

    # We repurpose the "transcript" section to show the user the extracted description
    display_transcript = description if description else "(No written description found on this post. Context extracted directly from video/audio visual analysis.)"

    return AnalyzeResponse(
        url=req.url,
        transcript=display_transcript,
        destinations=[Destination(**d) for d in destinations],
    )
