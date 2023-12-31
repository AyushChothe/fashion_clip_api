FROM python:3.10.12-slim-bullseye

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt install build-essential -y

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python","main.py" ]