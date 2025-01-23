# Base image
FROM python:3.10.9 AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs configs/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

# Expose the port for FastAPI
EXPOSE 8000

# Set default entrypoint for Python script
ENTRYPOINT ["uvicorn"]

# ENTRYPOINT ["python", "-u", "main.py"]

# Command to start FastAPI and trigger training
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000", "&", "sleep", "5", "&&", "curl", "-X", "POST", "http://127.0.0.1:8000/train"]
