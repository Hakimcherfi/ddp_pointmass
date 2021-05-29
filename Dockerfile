FROM python:3.9.5-slim

RUN pip install numpy==1.20.3 matplotlib==3.4.2

WORKDIR /ddp

COPY ./ddp.py .

ENTRYPOINT ["python3","ddp.py"]
