FROM python:3.7-buster
WORKDIR /app
COPY requirements.txt .
COPY app.py .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt