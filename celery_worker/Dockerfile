FROM python:3.12.4

#WORKDIR /usr/src/app/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["celery", "-A", "app.celery", "worker", "--loglevel=info"]
