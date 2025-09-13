FROM python:3.11-slim

# system deps for numpy/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code + model into image
COPY app/ /app/

# serve on 8000 inside the container
EXPOSE 8000
ENV PORT=8000

# run Dash via gunicorn (Dash exposes Flask server = `server` in app/app.py)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:server"]
