# Use a Python base image
FROM python:3.10.9

# Install required dependencies, including curl and wget for gsutil installation
RUN apt-get update && apt-get install -y curl wget && \
    apt-get clean

# Install Google Cloud SDK for gsutil
RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-426.0.0-linux-x86_64.tar.gz && \
    tar -xzf google-cloud-sdk-426.0.0-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh --quiet && \
    rm -rf google-cloud-sdk-426.0.0-linux-x86_64.tar.gz

# Set PATH for gcloud and gsutil
ENV PATH="/google-cloud-sdk/bin:$PATH"

# Set working directory
WORKDIR /app

# Ensure necessary directories exist
RUN mkdir -p /app/src /app/configs

# Copy all necessary files from the public bucket
RUN gsutil -m cp -r gs://mlops-bucket-224229-1/src/* /app/src/ && \
    gsutil -m cp -r gs://mlops-bucket-224229-1/configs/* /app/configs/ && \
    gsutil -m cp gs://mlops-bucket-224229-1/main.py /app/main.py && \
    gsutil -m cp gs://mlops-bucket-224229-1/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Default command to run the application
CMD ["/usr/local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

