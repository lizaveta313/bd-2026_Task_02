FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy pyarrow

COPY calc_data.py /app/calc_data.py

ENTRYPOINT ["python3", "calc_data.py"]
