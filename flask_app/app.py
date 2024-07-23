from flask import Flask,jsonify,request,render_template
from celery import Celery
from celery import group
from celery import Task
import json
import os
import redis

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL') # must contain password in URL for remote
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND')
app.config['REDIS_HOST'] = os.environ.get('REDIS_HOST')
app.config['REDIS_PASS'] = os.environ.get('REDIS_PASS')
app.config['REDIS_STATE'] = os.environ.get('REDIS_STATE')


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)
if app.config['REDIS_STATE'] == "local": # mean it's being run locally
    redis_client = redis.Redis(host=app.config['REDIS_HOST'], port=6379, db=0,decode_responses=True)#,
                                 #username="default", password=app.config['REDIS_PASS'])
else: # otherwise it should be remote
    redis_client = redis.Redis(host=app.config['REDIS_HOST'], port=6379, db=0,decode_responses=True,
                                 username="default", password=app.config['REDIS_PASS'])

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/longtask')
def longtask():
    task = add_together.delay(23, 42)
    return f'Task ID: {task.id}'

@app.route('/gputask')
def gputask():
    task = gpu_test.delay() # right now it's GPU test but we can choose other tasks..
    # If user will supply parameters from web interface, that will need to be processed here..
    return f'Task ID: {task.id}'

# @app.route('/remove_one')
# def remove_one():
#     results = []
#     task_count = 0
#     for value in range(0,10):
#         results.append(celery.send_task('tasks.helium_train',args=[task_count],queue="GPU_queue"))
#         task_count += 1
#     #task = helium_train_remove_one.delay()
#     return f'Task IDs: {[task.id for task in results]}'

@app.route('/remove_one')
def remove_one():
    task_count = 0
    tasks = []
    for value in range(0,10):
        tasks.append(celery.signature('tasks.helium_train',args=[task_count],options={"queue":"GPU_queue"}))
        task_count += 1
    #task = helium_train_remove_one.delay()
    job = group(tasks)
    results = job.apply_async()
    #results.join()
    return f'Task IDs: {[task.id for task in results]}'

#
@app.route('/result/<task_id>')
def get_result(task_id):
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
            #'status': task.info.get('status', '')
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/tasks')
def tasks():
    keys = redis_client.keys('celery-task-meta-*')
    tasks = []
    for key in keys:
        results = json.loads(redis_client.get(key)) # need to load result to a dictionary
        tasks.append(results)

    return render_template('tasks.html', tasks=tasks)

@celery.task(name='tasks.add_together')
def add_together(a, b):
    return a + b

@celery.task(name='tasks.gpu_test')
def gpu_test():
    return

# @celery.task(name='tasks.helium_train')
# def helium_train(data):
#     return
#
# @celery.task(name='tasks.helium_train_remove_one')
# def helium_train_remove_one():
#     data_chucks = [1,2,4,5,6,7,8]
#     job = group(helium_train.s(data) for data in data_chucks)
#     result = job.apply_async()
#     return result.get()

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
