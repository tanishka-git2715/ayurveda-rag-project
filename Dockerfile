# Use slim Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy data folder
COPY data ./data

# Expose FastAPI port
EXPOSE 7860

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]