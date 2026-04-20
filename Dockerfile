FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first (~200 MB vs ~2.5 GB for CUDA build) to avoid build timeouts
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
