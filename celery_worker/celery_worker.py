from celery import Celery
from celery import group
import os
import datetime

def make_celery():
    celery = Celery(
        'celery_worker',
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0'),
        task_queues={
            'log_queue': {
                'exchange': 'log_queue',
                'exchange_type': 'direct',
                'binding_key': 'log_queue',
            },
        }
    )
    print("making celery worker")

    return celery

celery = make_celery()


@celery.task(name='tasks.add_together')
def add_together(a, b):
    print("running add_together from celery_work tasks.py")
    return a + b

@celery.task(name='tasks.log_results',queue='log_queue',acks_late=True)
def log_results(results):
    print(f"writing {results}")
    filename = '/logs/' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S-") + str(results["results_type"]) + '.txt'
    with open(filename,'a') as file:
        file.write(str(results["data"])+"\n")
    return(results)

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