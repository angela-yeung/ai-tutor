FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the NLI model at build time so it's baked into the image
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
