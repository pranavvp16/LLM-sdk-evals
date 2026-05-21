FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sdk ./sdk
COPY services/api ./services/api
COPY scripts ./scripts

ENV PYTHONPATH=/app

EXPOSE 8000
