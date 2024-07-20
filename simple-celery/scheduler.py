
# based on code from https://vismay-t.medium.com/celery-on-multiple-servers-5e561b3b36d7

from celery import Celery
import time
import json

#broker url for redis
RBROKER="your_redis_url"
CELERY_BROKER_URL = f"redis://{RBROKER}:6379/0"

#initialize celery object
celery_obj = Celery('scheduler',broker=CELERY_BROKER_URL,)

#Declaring the queue
celery_obj.conf.task_routes = {
    "scheduler.create_task":{'queue':'create_task'}
}

# function we want to call
@celery_obj.task(bind=True, queue='create_task',name='create_task')
def create_task(self,task):
    # nothing needed here
    pass
