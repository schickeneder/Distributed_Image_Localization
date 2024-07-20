from scheduler import celery_obj

celery_obj.send_task("create_task",[function_arguments])
