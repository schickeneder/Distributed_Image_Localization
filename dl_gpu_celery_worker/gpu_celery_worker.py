from celery import Celery, chord, group
import os, sys
import redis
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

redis_state = os.environ.get('REDIS_STATE')
redis_host = os.environ.get('REDIS_HOST')
redis_pass = os.environ.get('REDIS_PASS')

if redis_state == "local": # mean it's being run locally
    redis_client = redis.Redis(host=redis_host, port=6379, db=0,decode_responses=True)#,
                                 #username="default", password=app.config['REDIS_PASS'])
else: # otherwise it should be remote
    redis_client = redis.Redis(host=redis_host, port=6379, db=0,decode_responses=True,
                               username="default", password=redis_pass)

def find_dataset(file_path):
    print(f"Looking for dataset in {file_path}")
    if os.path.exists(file_path):
        print(f"Found dataset in {file_path}")
        return True
    else:
        if not os.path.exists('datasets/generated'):
            os.makedirs('datasets/generated')
        redis_file_key = f'file:{file_path}'
        file_data = redis_client.get(redis_file_key)
        if file_data:
            # Save the file locally
            with open(file_path, 'w') as f:
                f.write(file_data)
            print(f"File saved to {file_path}.")
            return True
        else:
            print(f"No file found in Redis with key {redis_file_key}")
            return False



# Retrieves the list of rx_lats for all nodes in the dataset specified in params
# Populates rx_blacklist with this list for use in subsequent functions
@celery.task (name='tasks.get_rx_lats')
def get_rx_lats_task(params):
    # TODO: make sure dataset is present
    find_dataset(params['data_filename'])
    rx_lats = helium_training.get_rx_lats(params)
    print(f"****args for task.get_rx_lats: {params}")
    params['task_id'] = get_rx_lats_task.request.id
    return {**params,"rx_blacklist": rx_lats} # keep them there for purposes of passing args

@celery.task(name='tasks.get_time_splits')
def get_time_splits(params):
    # TODO: make sure dataset is present
    find_dataset(params['data_filename'])
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
#
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
    find_dataset(params['data_filename'])
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

# used to process and log a single run
@celery.task(name='tasks.train_one_and_log', acks_late=True)
def train_one_and_log(params):
    params = {**params,"results_type" : "default"}
    print(f"****args for tasks.train_one_and_log: {params}, logging results")
    find_dataset(params['data_filename']) # download from redis cache if not present
    res = helium_training.main_process(params)
    dict_results = {"results_type": "results", "params": params,"data": res}
    task1 = celery.signature("tasks.log_results", args=[dict_results], options={"queue": "log_queue"})
    task1.apply_async()
    # TODO: add the next task for all the percentiles results to get rx_blacklists
    return dict_results

