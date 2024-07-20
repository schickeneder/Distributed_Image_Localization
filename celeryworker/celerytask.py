from celery import Celery

app = Celery('task')
app.config_from_object('celeryconfig')

@app.task
def add_numbers():
    return
