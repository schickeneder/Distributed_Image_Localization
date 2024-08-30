from flask import Flask,jsonify,request,render_template,current_app,flash
from celery import Celery, group, chain, chord
import json
import os
import redis
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import math


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL') # must contain password in URL for remote
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND')
app.config['REDIS_HOST'] = os.environ.get('REDIS_HOST')
app.config['REDIS_PASS'] = os.environ.get('REDIS_PASS')
app.config['REDIS_STATE'] = os.environ.get('REDIS_STATE')
app.config['GROUP_JOBS'] = {} # make a dict to support other metadata like start time
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['csv']

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL'],
        result_expires=0,
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

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# TODO: load full helium coords list into memory and see how quickly it can filter out (in real time?) datasets based on coords


# store the global dataset in memory so we can quickly select subsets when requested
def load_global_dataset(filepath):
    global global_dataset_loaded
    global global_dataset
    try:
        dataset = pd.read_csv(filepath)
        column_names = dataset.columns.tolist()

        print(column_names)

        column_names[0] = 'time'
        column_names[1] = 'lat1'
        column_names[2] = 'lon1'
        column_names[3] = 'lat2'
        column_names[4] = 'lon2'
        column_names[5] = 'txpwr'
        column_names[6] = 'rxpwr'

        dataset.columns = column_names
        global_dataset_loaded = True
        global_dataset = dataset
    except Exception as e:
        print(f"Could not load dataset {filepath} because {e}")


# geofilter global dataset to produce regional subset
def filter_coordinates(df, coordinates):
    bottom_left, top_right = (coordinates)
    return df[
        (df['lat1'] >= bottom_left[0]) & (df['lat1'] <= top_right[0]) &
        (df['lon1'] >= bottom_left[1]) & (df['lon1'] <= top_right[1]) &
        (df['lat2'] >= bottom_left[0]) & (df['lat2'] <= top_right[0]) &
        (df['lon2'] >= bottom_left[1]) & (df['lon2'] <= top_right[1])
    ]

def cache_datafile(file_path):
    redis_file_key = f'file:{file_path}'
    expiration_time = 60 * 60 * 24 # 24 hours in seconds

    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
            redis_client.set(redis_file_key, file_data, ex=expiration_time)

            redis_client.set(file_path, file_data)
    except Exception as e:
        print(f"Encountered exception trying to cache datafile {e}")
        return False
    return True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# returns BL TR corner coordinates of a square centered at lat,lon
def get_square_corners(lat, lon, side_length):
    # Earth's radius in meters
    R = 6378137.0

    # Convert the side length from meters to degrees
    d_lat = (side_length / 2) / R * (180 / math.pi)
    d_lon = (side_length / 2) / (R * math.cos(math.pi * lat / 180)) * (180 / math.pi)

    # Bottom-left coordinates
    bottom_left_lat = lat - d_lat
    bottom_left_lon = lon - d_lon

    # Top-right coordinates
    top_right_lat = lat + d_lat
    top_right_lon = lon + d_lon

    return (bottom_left_lat, bottom_left_lon), (top_right_lat, top_right_lon)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/generate_datasets')
def generate_datasets():
    global global_dataset
    global global_dataset_loaded

    square_length = 8000 # 8000 meters ~ 5 miles

    cities_data_50 = [
        ("New York City, NY", "NYC", 40.7128, -74.0060),
        ("Los Angeles, CA", "LA", 34.0522, -118.2437),
        ("Chicago, IL", "CHI", 41.8781, -87.6298),
        ("Houston, TX", "HOU", 29.7604, -95.3698),
        ("Phoenix, AZ", "PHX", 33.4484, -112.0740),
        ("Philadelphia, PA", "PHL", 39.9526, -75.1652),
        ("San Antonio, TX", "SAT", 29.4241, -98.4936),
        ("San Diego, CA", "SD", 32.7157, -117.1611),
        ("Dallas, TX", "DAL", 32.7767, -96.7970),
        ("San Jose, CA", "SJ", 37.3382, -121.8863),
        ("Austin, TX", "AUS", 30.2672, -97.7431),
        ("Jacksonville, FL", "JAX", 30.3322, -81.6557),
        ("Fort Worth, TX", "FTW", 32.7555, -97.3308),
        ("Columbus, OH", "COL", 39.9612, -82.9988),
        ("Charlotte, NC", "CLT", 35.2271, -80.8431),
        ("San Francisco, CA", "SF", 37.7749, -122.4194),
        ("Indianapolis, IN", "IND", 39.7684, -86.1581),
        ("Seattle, WA", "SEA", 47.6062, -122.3321),
        ("Denver, CO", "DEN", 39.7392, -104.9903),
        ("Washington, DC", "DC", 38.9072, -77.0369),
        ("Boston, MA", "BOS", 42.3601, -71.0589),
        ("El Paso, TX", "ELP", 31.7619, -106.4850),
        ("Nashville, TN", "NSH", 36.1627, -86.7816),
        ("Detroit, MI", "DET", 42.3314, -83.0458),
        ("Oklahoma City, OK", "OKC", 35.4676, -97.5164),
        ("Portland, OR", "PDX", 45.5152, -122.6784),
        ("Las Vegas, NV", "LV", 36.1699, -115.1398),
        ("Memphis, TN", "MEM", 35.1495, -90.0490),
        ("Louisville, KY", "LOU", 38.2527, -85.7585),
        ("Baltimore, MD", "BAL", 39.2904, -76.6122),
        ("Milwaukee, WI", "MKE", 43.0389, -87.9065),
        ("Albuquerque, NM", "ABQ", 35.0844, -106.6504),
        ("Tucson, AZ", "TUC", 32.2226, -110.9747),
        ("Fresno, CA", "FRE", 36.7378, -119.7871),
        ("Mesa, AZ", "MES", 33.4152, -111.8315),
        ("Sacramento, CA", "SAC", 38.5816, -121.4944),
        ("Atlanta, GA", "ATL", 33.7490, -84.3880),
        ("Kansas City, MO", "KC", 39.0997, -94.5786),
        ("Colorado Springs, CO", "COS", 38.8339, -104.8214),
        ("Miami, FL", "MIA", 25.7617, -80.1918),
        ("Raleigh, NC", "RAL", 35.7796, -78.6382),
        ("Omaha, NE", "OMA", 41.2565, -95.9345),
        ("Long Beach, CA", "LB", 33.7701, -118.1937),
        ("Virginia Beach, VA", "VB", 36.8529, -75.9780),
        ("Oakland, CA", "OAK", 37.8044, -122.2711),
        ("Minneapolis, MN", "MIN", 44.9778, -93.2650),
        ("Tulsa, OK", "TUL", 36.1540, -95.9928),
        ("Tampa, FL", "TPA", 27.9506, -82.4572),
        ("Arlington, TX", "ARL", 32.7357, -97.1081),
        ("New Orleans, LA", "NO", 29.9511, -90.0715)
    ]

    params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
              'func_list': ["MSE","COM","EMD"], "data_filename": "",
              "results_type": "default", "coordinates" : [(47.556372, -122.360229), (47.63509, -122.281609)]}

    if global_dataset_loaded:
        for row in cities_data_50:
            lat,lon = row[-2:]
            bl_coords, tr_coords = get_square_corners(lat,lon,square_length)
            local_dataset = filter_coordinates(global_dataset,
                                               ((float(bl_coords[0]), float(bl_coords[1])),
                                                (float(tr_coords[0]), float(tr_coords[1]))))
            data_filename = "datasets/" + str(row[1]) + str(square_length/1000) + '.csv' # <city abbrev><square_length in km>
            local_dataset.to_csv(data_filename, index=False)
            cache_datafile(params["data_filename"])

            task1 = celery.signature("tasks.train_one_and_log",
                                     args=[{**params,"data_filename": data_filename,
                                            "coordinates": [bl_coords, tr_coords]}],
                                     options={"queue": "gpu_queue"})
            task1.apply_async()

        return "Processed cities list and start tasks."


    else:
        return "Waiting for global_dataset to load.."
# @app.route('/upload_file', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If user does not select a file, the browser submits an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = file.filename
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             flash('File successfully uploaded')
#             return redirect(url_for('process_file', filename=filename))
#     return render_template('index.html')

@app.route('/get_dataset/<lat1>/<lon1>/<lat2>/<lon2>')
def get_dataset(lat1, lon1, lat2, lon2):
    global global_dataset
    global global_dataset_loaded

    if global_dataset_loaded:
        local_dataset = filter_coordinates(global_dataset,
                                           ((float(lat1), float(lon1)), (float(lat2), float(lon2))))
        #local_dataset.to_csv(outfile_name, index=False)

        return local_dataset.to_string()
    else:
        return "Waiting for global_dataset to load.."

@app.route('/longtask')
def longtask():
    task = add_together.delay(23, 42)
    return f'Task ID: {task.id}'

@app.route('/data_files', methods=['GET'])
def data_files():
    res = [
    "datasets/helium_SD/SF30_helium.csv",
    "datasets/helium_SD/SF31_helium.csv",
    "datasets/helium_SD/SF32_helium.csv"
    ]

    return jsonify(res)

@app.route('/gputask')
def gputask():
    task = gpu_test.delay() # right now it's GPU test but we can choose other tasks..
    # If user will supply parameters from web interface, that will need to be processed here..
    return f'Task ID: {task.id}'

@app.route('/logtask')
def logtask():
    task = log_test.delay() # right now it's GPU test but we can choose other tasks..
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

    # # for test
    # params = {"max_num_epochs": 1, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE"], "data_filename": "datasets/helium_SD/SD30_helium.csv",
    #           "results_type": "remove_one", "coordinates" : [(32.732419, -117.247639),(32.796799, -117.160606)]}

    # params = {"max_num_epochs": 20, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE","COM","EMD"], "data_filename": "datasets/helium_SD/SD30_driving.csv",
    #           "results_type": "remove_one", "coordinates" : [(32.732419, -117.247639),(32.796799, -117.160606)]}

    # SF dataset spans from Aug 2017 to April 2024, but mostly from 1635890982 11/2/21 to 1681831899 4/18/23
    # choose about halfway through: 1654394803 6/5/22
    # that was too much, so now we choose from 1671622676 12/21/22 1675273138

    # params = {"max_num_epochs": 10, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE","COM","EMD"], "data_filename": "datasets/helium_SD/SF30_helium.csv",
    #           "timespan": 1675273138, "results_type": "remove_one",
    #           "coordinates" : [(37.610424, -122.531204),(37.808156, -122.336884)]}

    params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
              'func_list': ["MSE","COM","EMD"], "data_filename": "datasets/helium_SD/SEA30_helium.csv",
              "results_type": "remove_one_results", "coordinates" : [(47.556372, -122.360229), (47.63509, -122.281609)]}

    cache_datafile(params["data_filename"])

    task1 = celery.signature("tasks.get_rx_lats",args=[params],options={"queue":"GPU_queue"})
    task2 = celery.signature("tasks.split_and_group",options={"queue":"GPU_queue"})
    task3 = celery.signature("tasks.log_results", options={"queue": "log_queue"})

    # this logs task_id but that actual results come through split_and_group task

    result = chain(task1,task2,task3).apply_async()

    print(result)
    return f'workflow result: {result}'

# This function trains models at different consecutive spans of time for one location
# the purpose is to observe changes in performance due to nodes and protocols over time
# For each model it will track the timespan tested and the number of nodes present
@app.route('/split_time_span')
def split_time_span():
    span = 2628288 # one month at a time seems like it should be good enough to see changes, but not too much compute


    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0], "splits": [],
    #           'func_list': ["MSE","COM","EMD"], "data_filename": "datasets/helium_SD/SEA30_helium.csv",
    #           "split_timespan": span, "results_type": "split_timespan_results",
    #           "coordinates" : [(47.556372, -122.360229), (47.63509, -122.281609)]}

    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],"splits": [],
    #           'func_list': ["MSE","COM","EMD"], "data_filename": "datasets/helium_SD/SF30_helium.csv",
    #           "split_timespan": span, "results_type": "split_timespan_results",
    #           "coordinates" : [(37.610424, -122.531204),(37.808156, -122.336884)]}

    params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],"splits": [],
              'func_list': ["MSE","COM","EMD"], "data_filename": "datasets/helium_SD/SD30_helium.csv",
              "split_timespan": span, "results_type": "split_timespan_results",
              "coordinates" : [(32.732419, -117.247639),(32.796799, -117.160606)]}

    cache_datafile(params["data_filename"])


    # task1 returns the timestamps of the samples corresponding to each split and stores ind "splits" list
    task1 = celery.signature("tasks.get_time_splits",args=[params],options={"queue":"GPU_queue"})
    task2 = celery.signature("tasks.split_and_group_timespan",options={"queue":"GPU_queue"})
    task3 = celery.signature("tasks.log_results", options={"queue": "log_queue"})

    # this logs task_id but that actual results come through split_and_group task

    result = chain(task1,task2,task3).apply_async()

    print(result)
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

@celery.task(name='tasks.gpu_test',queue='GPU_queue')
def gpu_test():
    return

@celery.task(name='tasks.group_remove_one')
def group_remove_one(results):
    return

@celery.task(name='tasks.log_results',queue='log_queue')
def log_results(results):
    print(f"writing {results}")
    with open('/logs/log_results.txt','a') as file:
        file.write(str(results))
    return(results)


if __name__ == '__main__':
    filepath = "/datasets/global/all_data.csv"
    print("Loading global dataset..",flush=True) # print immediately
    global_dataset_loaded = False # flag updates when global_dataset load completes
    global_dataset = None # placeholder for global variable
    threading.Thread(target=load_global_dataset, args=(filepath,)).start()
    app.run(host='0.0.0.0',debug=True)
