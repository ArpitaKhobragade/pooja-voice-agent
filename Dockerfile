# Pooja web demo. Build context is the project root:
#   docker build -t pooja-web .
#   docker run -p 8000:8000 --env-file backend/.env pooja-web
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POOJA_VAD_BACKEND=webrtcvad

WORKDIR /app/backend

COPY backend/requirements-web.txt ./requirements-web.txt
RUN pip install --no-cache-dir -r requirements-web.txt

COPY backend/ ./

# Hosts like Render set PORT; web_server.py reads it (default 8000).
EXPOSE 8000
CMD ["python", "web_server.py"]
