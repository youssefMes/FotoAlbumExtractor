FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update

RUN apt install -y libgl1-mesa-glx libglib2.0-0

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "main.py"]
