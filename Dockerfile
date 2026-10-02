FROM python:3.11-slim

WORKDIR /

RUN pip install --no-cache-dir runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]