FROM python:3.11

# 1. Non-root user for security
RUN useradd -m -u 1000 user

WORKDIR /app

# 2. Install system dependencies (ffmpeg is still required for audio/video processing)
RUN apt-get update && apt-get install -y build-essential ffmpeg curl git && rm -rf /var/lib/apt/lists/*

# 3. Upgrade core python tools FIRST in a completely separate layer.
RUN pip install --no-cache-dir --upgrade pip "setuptools<70.0.0" wheel

# 4. Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# 5. Copy all files into the container, assigning ownership to our new user
COPY --chown=user . .

# 6. Move index.html into frontend folder if it was accidentally uploaded to the root
RUN mkdir -p frontend && if [ -f "index.html" ]; then mv index.html frontend/; fi && chown -R user:user frontend

# 7. Redirect cache to /tmp (always writable)
ENV XDG_CACHE_HOME=/tmp/.cache \
    WHISPER_CACHE=/tmp/.cache/whisper
RUN mkdir -p /tmp/.cache/whisper && chmod -R 777 /tmp/.cache

# 8. Switch to the non-root user for security
USER user

# MAGIC FIX FOR RENDER: Use the dynamic $PORT environment variable assigned by Render!
# (We set a default fallback to 7860 just in case)
ENV PORT=7860
EXPOSE $PORT

# Use shell form of CMD (no brackets) so the $PORT variable actually gets evaluated!
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
