from celery import Celery
import os
import subprocess

def make_celery():
    celery = Celery(
        'celery_worker',
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    )
    print("making celery worker")
    return celery

celery = make_celery()

@celery.task(name='tasks.add_together')
def add_together(a, b):
    print("running add_together from celery_work tasks.py")
    return a + b

@celery.task(name='tasks.gpu_test')
def gpu_test():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print('stdout:', result.stdout)
    print('stderr:', result.stderr)
    print('Return code:', result.returncode)
    return result.stdout

@celery.task(name='tasks.helium_train')
def helium_train(data):
    return data
