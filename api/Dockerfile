FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
WORKDIR /app
COPY requirements.txt requirements.txt
COPY app.py app.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=TRUE