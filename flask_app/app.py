from flask import Flask,jsonify,request,render_template,current_app
from celery import Celery, group, chain, chord
import json
import os
import redis
import hashlib
from datetime import datetime


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL') # must contain password in URL for remote
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND')
app.config['REDIS_HOST'] = os.environ.get('REDIS_HOST')
app.config['REDIS_PASS'] = os.environ.get('REDIS_PASS')
app.config['REDIS_STATE'] = os.environ.get('REDIS_STATE')
app.config['GROUP_JOBS'] = {} # make a dict to support other metadata like start time


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
    task_ids = job.apply_async()
    checksum = hashlib.sha1(''.join(str(task_ids)).encode('utf-8')).hexdigest()[:16]
    current_app.config['GROUP_JOBS'][checksum] = {"group" : task_ids,"start_time" : datetime.now().isoformat()}
    task_str = [str(task) for task in task_ids]
    return f'Task IDs: {task_str} {current_app.config['GROUP_JOBS'][checksum]["start_time"]}'

@app.route('/remove_one2')
def remove_one2():

    params = {"max_num_epochs": 2, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [],
              'func_list': ["MSE"], "data_filename": "datasets/helium_SD/filtered_Seattle_data.csv",
              "results_type": "remove_one"}

    task1 = celery.signature("tasks.get_rx_lats",args=[params],options={"queue":"GPU_queue"})
    task2 = celery.signature('tasks.split_and_group',options={"queue":"GPU_queue"})

    result = chain(task1,task2).apply_async()

    return f'workflow result: {result}'

@app.route('/recent_group_status')
def recent_group_status():
    groups = current_app.config['GROUP_JOBS']
    completion_counts = {}
    for group in groups.keys():
        #checksum = hashlib.sha1(''.join(str(group)).encode('utf-8')).hexdigest()[:16]
        success_count = 0
        for task in groups[group]["group"]:
            if task.status == 'SUCCESS':
                success_count += 1
        completion_counts[str(group)] = f"{success_count}/{len(group)}"

    #checksum = hashlib.sha1(''.join(str(recent_groups)).encode('utf-8')).hexdigest()[:16]
    #return (f"Recent Group: {checksum}: {success_count}/{len(recent_groups)} Complete")
    return render_template('recent_group_status.html', completion_counts=completion_counts, groups=groups)
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

@app.route('/interactive_map')
def interactive_map():


    return render_template('interactive_map.html')


@celery.task(name='tasks.add_together')
def add_together(a, b):
    return a + b

@celery.task(name='tasks.gpu_test')
def gpu_test():
    return

@celery.task(name='tasks.group_remove_one')
def group_remove_one(results):
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
