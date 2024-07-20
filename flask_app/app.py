from flask import Flask
from celery import Celery
import os

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/longtask')
def longtask():
    task = add_together.delay(23, 42)
    return f'Task ID: {task.id}'

@celery.task
def add_together(a, b):
    return a + b

if __name__ == '__main__':
    app.run(host='0.0.0.0')
