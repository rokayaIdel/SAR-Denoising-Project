FROM python:3.11-slim

WORKDIR /app

# dépendances système
RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY etl_run.py .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "etl_run.py"]

