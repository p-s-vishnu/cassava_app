FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
WORKDIR /app
COPY requirements.txt requirements.txt
COPY app.py app.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=TRUE
# EXPOSE 8003
# ENTRYPOINT gunicorn app:app --bind 0.0.0.0:8003 --timeout 1200 -w 1 -k uvicorn.workers.UvicornWorker