FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/train.py .
COPY src/model.py .

CMD ["python", "train.py"]
