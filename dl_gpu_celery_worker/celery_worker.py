from celery import Celery
import os, sys
import subprocess
import time
import random
#from dl_img_loc_lite import helium_training
from celery import chord, group

def make_celery():
    celery = Celery(
        'celery_worker',
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    )
    print("making celery worker")
    return celery

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, 'dl_img_loc_lite'))
#os.chdir('dl_img_loc_lite')
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

@celery.task (name='tasks.get_rx_lats')
def get_rx_lats_task(params):
    #rx_lats = helium_training.get_rx_lats(params)
    print(f"****args for task.get_rx_lats: {params}")
    rx_lats = [1,2,3]
    return rx_lats

@celery.task(name='tasks.split_and_group')
def split_and_group_rx_lats(rx_lats):
    print(f"****args for tasks.split_and_group_rx_lats: {rx_lats}")
    g = group(group_remove_one2.s(rx_lat).set(queue="GPU_queue") for rx_lat in rx_lats[:3])
    res = g()
    print(f" res {res} ")
    return 2

@celery.task(name='tasks.group_remove_one2')
def group_remove_one2(params):
    print(f"****args for tasks.group_remove_one2: {params}")
    return 3

@celery.task(name='tasks.process_remove_one_results')
def process_remove_one_results(results):
    print("running process_remove_one_results")
    return f"all done and results {results}"

