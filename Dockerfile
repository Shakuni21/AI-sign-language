FROM python:3.13-slim

# Install Linux libraries required by MediaPipe/OpenGL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libglib2.0-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY index.html .
COPY sign_model.pkl .
COPY hand_landmarker.task .

# Render provides the PORT environment variable
EXPOSE 10000

# Start FastAPI
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}