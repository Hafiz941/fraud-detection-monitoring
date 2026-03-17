# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirement file first (better caching)
COPY requirements.txt .

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip

# Install numpy first to ensure binary compatibility for compiled packages
RUN pip install --no-cache-dir "numpy==1.24.4"

# Install the rest of the requirements (scikit-learn will match numpy)
RUN pip install --no-cache-dir -r requirements.txt

# Install the pandas
RUN pip install pandas

# Copy project files into the container
COPY . /app

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]