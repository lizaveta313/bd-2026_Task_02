FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir polars==1.41.2

COPY calc_data.py /app/calc_data.py

ENTRYPOINT ["python3", "/app/calc_data.py"]
