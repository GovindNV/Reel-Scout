FROM python:3.11

# 1. HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# 2. Install system dependencies (ffmpeg is still required for Whisper's audio processing)
RUN apt-get update && apt-get install -y build-essential ffmpeg curl git && rm -rf /var/lib/apt/lists/*

# 3. Upgrade core python tools FIRST in a completely separate layer.
# Pin setuptools to <70 to prevent the 'pkg_resources' ModuleNotFoundError!
RUN pip install --no-cache-dir --upgrade pip "setuptools<70.0.0" wheel

# 4. Copy requirements and install them
COPY --chown=user requirements.txt .
# Use --no-build-isolation so pip uses our downgraded setuptools
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# 5. Copy all files into the container, assigning ownership to our new user
COPY --chown=user . .

# 6. Move index.html into frontend folder if it was accidentally uploaded to the root
RUN mkdir -p frontend && if [ -f "index.html" ]; then mv index.html frontend/; fi && chown -R user:user frontend

# 7. Claude's excellent suggestion: Redirect cache to /tmp (always writable in HF Spaces)
ENV XDG_CACHE_HOME=/tmp/.cache \
    WHISPER_CACHE=/tmp/.cache/whisper
RUN mkdir -p /tmp/.cache/whisper && chmod -R 777 /tmp/.cache

# 8. Switch to the non-root user for security and HF compliance
USER user

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]