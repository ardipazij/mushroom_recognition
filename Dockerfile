FROM python:3.9

RUN apt-get update
RUN apt install -y  nginx
RUN pip install --no-cache-dir python-telegram-bot

WORKDIR /app
COPY . .
RUN cp my_flask_app.conf /etc/nginx/sites-available/
RUN ln -s /etc/nginx/sites-available/my_flask_app.conf /etc/nginx/sites-enabled/
RUN nginx -t
RUN pip install -r requirements.txt
RUN chmod +x multiprocess.sh

CMD ["./multiprocess.sh"]
