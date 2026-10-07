FROM python:3.6-slim

WORKDIR /app

COPY requirements-train.txt .

RUN pip install --trusted-host --no-cache-dir -r requirements-train.txt

COPY main-train.py .

ENV MODEL_PATH=model.pkl

CMD ["python", "main-train.py"]

