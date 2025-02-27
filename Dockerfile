FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY transcribe.py /app/

RUN chmod +x /app/transcribe.py

ENTRYPOINT ["python", "/app/transcribe.py"]
