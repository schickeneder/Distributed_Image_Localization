FROM python:3.9

#WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["celery", "-A", "celery_worker.celery", "worker", "--loglevel=info"]
