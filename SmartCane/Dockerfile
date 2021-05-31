FROM python:3.7

RUN mkdir /docker-server
WORKDIR /docker-server

ADD requirements.txt /docker-server/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.4.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN apt update
RUN apt-get -y install libgl1-mesa-glx
RUN pip install opencv-python
RUN pip install django
RUN pip install djangorestframework
RUN pip install matplotlib
RUN pip install ipython
RUN pip install gunicorn

ADD . /docker-server/

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
