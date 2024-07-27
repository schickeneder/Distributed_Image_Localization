from celery import Celery, chord, group
import os, sys
import subprocess
import time
import random
import helium_training


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

@celery.task(name='tasks.get_rx_lats_from_ds9')
def get_rx_lats_from_ds9(data):
    time.sleep(random.randrange(0,15))
    return data


@celery.task(name='tasks.helium_train')
def helium_train(data):
    time.sleep(random.randrange(0,15))
    return data

# Retrieves the list of rx_lats for all nodes in the dataset specified in params
# Populates rx_blacklist with this list for use in subsequent functions
@celery.task (name='tasks.get_rx_lats')
def get_rx_lats_task(params):
    rx_lats = helium_training.get_rx_lats(params)
    print(f"****args for task.get_rx_lats: {params}")
    return {**params,"rx_blacklist": rx_lats} # keep them there for purposes of passing args

@celery.task(name='tasks.split_and_group')
def split_and_group_rx_lats(params):
    print(f"****args for tasks.split_and_group_rx_lats: {params['rx_blacklist']}")
    # create a group of tasks with each one being passed one of the rx_lats as the only member of rx_blacklist
    g = group(group_remove_one2.s({**params,"rx_blacklist" : [rx_lat]}).
              set(queue="GPU_queue") for rx_lat in params["rx_blacklist"][:3]) # TODO: eventually get rid of [:3]
    res = chord(g)(process_remove_one_results.s().set(queue="GPU_queue"))
    # No need to return anything because chord callback (process_remove_one_results) will when parallel tasks complete
    return "completed split_and_group_rx_lats"

@celery.task(name='tasks.group_remove_one2')
def group_remove_one2(params):
    print(f"****args for tasks.group_remove_one2: {params}")
    res = helium_training.main_process(params)
    #time.sleep(random.randrange(0,15))
    return res

@celery.task(name='tasks.process_remove_one_results')
def process_remove_one_results(params):
    print(f"****args for tasks.process_remove_one_results: {params}")
    return params

