FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY src/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]