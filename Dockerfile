FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py .
COPY serving.py .

# Run the web service on container startup.
CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8080"]
