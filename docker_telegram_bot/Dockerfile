FROM python:3.9

RUN apt-get update
RUN apt install -y  nginx
RUN pip install --no-cache-dir python-telegram-bot

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN chmod +x multiprocess.sh

CMD ["./multiprocess.sh"]
