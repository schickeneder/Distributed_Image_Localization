FROM python:3.9

#WORKDIR /usr/src/app


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#ENV PYTHONPATH="/usr/src/app"

COPY .. .

CMD ["celery", "-A", "gpu_celery_worker.celery", "worker", "--loglevel=debug","--concurrency=1","-Q","GPU_queue"]
