from celery import Celery, chord, group
import os, sys
import subprocess
import time
import random
import helium_training


def make_celery():
    celery = Celery(
        'gpu_celery_worker',
        backend=os.environ.get('REDIS_URL', 'redis://redis:6379/0'),
        broker=os.environ.get('REDIS_URL', 'redis://redis:6379/0'),
        result_expires=0,
        worker_prefetch_multiplier=1, # only allow workers to take 1 task at a time
        task_acks_late=False, # this is the global default, can enable for individual tasks with "acks_late=True"
        worker_concurrency=1, # this may be redundant if already included in the run instruction
        task_queues= {
            'GPU_queue': {
                'exchange': 'GPU_queue',
                'exchange_type': 'direct',
                'binding_key': 'GPU_queue',
            },
        }
    )
    print("making celery worker")
    return celery

celery = make_celery()

# Retrieves the list of rx_lats for all nodes in the dataset specified in params
# Populates rx_blacklist with this list for use in subsequent functions
@celery.task (name='tasks.get_rx_lats')
def get_rx_lats_task(params):
    rx_lats = helium_training.get_rx_lats(params) # TODO for debug, limit rx_lats to 3
    print(f"****args for task.get_rx_lats: {params}")
    params['task_id'] = get_rx_lats_task.request.id
    return {**params,"rx_blacklist": rx_lats} # keep them there for purposes of passing args

@celery.task(name='tasks.get_time_splits')
def get_time_splits(params):
    time_splits = helium_training.get_time_splits(params)
    print(f"****args for tasks.get_time_splits: {params}")
    return {**params,"splits": time_splits}

@celery.task(name='tasks.split_and_group')
def split_and_group_rx_lats(params):
    print(f"****args for tasks.split_and_group_rx_lats: {params['rx_blacklist']}")
    # create a group of tasks with each one being passed one of the rx_lats as the only member of rx_blacklist
    g = group(group_remove_one2.s({**params,"rx_blacklist" : [rx_lat]}).
              set(queue="GPU_queue") for rx_lat in params["rx_blacklist"])
    res = chord(g)(process_group_results.s().set(queue="GPU_queue"))
    # No need to return anything because chord callback (process_remove_one_results) will when parallel tasks complete
    return {"results_type": "params", "data": params}

@celery.task(name='tasks.split_and_group_timespan')
def split_and_group_timespan(params):
    print(f"****args for tasks.split_and_group_time_span: {params['splits']}")
    # create a group of tasks with each one being passed one of the rx_lats as the only member of rx_blacklist
    g = group(group_split_timespans.s({**params,"timespan" : timespan}).
              set(queue="GPU_queue") for timespan in params["splits"])
    res = chord(g)(process_group_results.s().set(queue="GPU_queue"))
    # No need to return anything because chord callback (process_remove_one_results) will when parallel tasks complete
    return {"results_type": "params", "data": params}


# Added acks_late so acknowledgement occurs after completion, with a time limit of 20 min
# this will re-queue the task if a worker is interrupted (e.g. a vast machine is outbid and removed during processing)
@celery.task(name='tasks.group_remove_one2',acks_late=True, time_limit=2400)
def group_remove_one2(params):
    params = {**params,"results_type" : "remove_one_results"}
    print(f"****args for tasks.group_remove_one2: {params}")
    res = helium_training.main_process(params)
    #time.sleep(random.randrange(0,15))
    return res

@celery.task(name='tasks.group_split_timespans',acks_late=True, time_limit=2400)
def group_split_timespans(params):
    params = {**params,"results_type" : "split_timespan_results"}
    print(f"****args for tasks.group_split_timespans: {params}")
    res = helium_training.main_process(params)
    #time.sleep(random.randrange(0,15))
    return res

@celery.task(name='tasks.process_group_results', acks_late=True)
def process_group_results(list_results):
    print(f"****args for tasks.process_group_results: {list_results}, logging results")
    # can do whatever other processing we want with the results here, but we will also log them.
    dict_results = {"results_type": "results", "data": list_results}
    task1 = celery.signature("tasks.log_results", args=[dict_results], options={"queue": "log_queue"})
    task1.apply_async()
    # TODO: add the next task for all the percentiles results to get rx_blacklists
    return dict_results

