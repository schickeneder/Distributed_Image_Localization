from celery import Celery
import os

def make_celery():
    celery = Celery(
        'tasks',
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
    )
    return celery

celery = make_celery()
