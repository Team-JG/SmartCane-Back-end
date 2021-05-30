FROM python:3.8
ENV PYTHONUNBUFFERED=1
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN apt-get update \
    && apt-get install -y \
    python3 python3-pip python3-dev python3-venv build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
RUN pip install django
RUN pip install djangorestframework
RUN pip install matplotlib
RUN pip install IPython
RUN apt update
RUN pip install opencv-python
RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl
RUN apt-get -y install libgl1-mesa-glx

RUN chmod +x /app/run.sh 

EXPOSE 8000

ENTRYPOINT ["/app/run.sh"]