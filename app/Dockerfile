# Streamlit demo image: packages the app and a trained model, so
# `docker run` serves predictions without any retraining.
#
# Build from the REPOSITORY ROOT (the build context needs src/ and models/):
#   docker build -f app/Dockerfile -t plant-disease-app .
#   docker run --rm -p 8501:8501 plant-disease-app
#
# models/ is gitignored, so a cloned repo has to train (or copy in) a model
# before the image can be built - see the README.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# libgl / libglib are needed by the imaging stack in a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so they stay cached when only the app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Inference needs the app, the shared preprocessing/Grad-CAM code, and the
# trained model + class_names.json.
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
