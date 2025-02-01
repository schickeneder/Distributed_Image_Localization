from flask import Flask,jsonify,request,render_template,current_app,flash, Response
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
import csv
import pickle
import io

from flask_cors import CORS # need this to allow cross origin requests


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL') # must contain password in URL for remote
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND')
app.config['REDIS_HOST'] = os.environ.get('REDIS_HOST')
app.config['REDIS_PASS'] = os.environ.get('REDIS_PASS')
app.config['REDIS_STATE'] = os.environ.get('REDIS_STATE')
app.config['GROUP_JOBS'] = {} # make a dict to support other metadata like start time
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['csv']
CORS(app)

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
        print("Done loading global dataset")
    except Exception as e:
        print(f"Could not load dataset {filepath} because {e}")

def load_dataset_from_csv(filepath):
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
        return dataset
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

def filter_deny_lat(lat1,lat2):
    with open('datasets/deny_lat_list.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        result = next(reader)
        filtered_lats = [lat for lat in result if float(lat1) < float(lat) < float(lat2)]
        return filtered_lats

def get_from_global_dataset(lat1, lon1, lat2, lon2):
    global global_dataset
    global global_dataset_loaded

    if global_dataset_loaded:
        local_dataset = filter_coordinates(global_dataset,
                                           ((float(lat1), float(lon1)), (float(lat2), float(lon2))))
        #local_dataset.to_csv(outfile_name, index=False)

        return local_dataset.to_string()
    else:
        return None


@app.route('/')
def hello_world():
    return 'Hello, World!'

# this one doesn't work yet..
@app.route('/run_one_model_corners/<dataset_filename>/<lat1>/<lon1>/<lat2>/<lon2>')
def run_one_model_corners(dataset_filename,lat1,lon1,lat2,lon2):
    global global_dataset_loaded
    print(f"Processing Dataset: {dataset_filename} with coords: {lat1,lon1,lat2,lon2}")

    params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
              'func_list': ["MSE","COM","EMD"], "data_filename": "",
              "results_type": "default", "coordinates" : [(lat1,lon1),(lat2,lon2)]}


    local_dataset = get_from_global_dataset(lat1,lon1,lat2,lon2)

    if local_dataset is None:
        print("dataset not available")
        return (f"Dataset not available, global dataset loaded: {global_dataset_loaded},"
            f" could not process {dataset_filename} at {lat1,lon1,lat2,lon2}")
    else:
        print("local_dataset loaded")

    try:
        local_dataset.to_csv(dataset_filename, index=False)
    except Exception as e:
        print(f"Couldn't save local dataset {dataset_filename} because {e}")
    cache_datafile(dataset_filename)


    rx_blacklist = [0]

    task1 = celery.signature("tasks.train_one_and_log",
                             args=[{**params,"data_filename": dataset_filename,
                                    "rx_blacklist": rx_blacklist,
                                    "coordinates": [(lat1,lon1),(lat2,lon2)]}],
                             options={"queue": "GPU_queue"})
    task1.apply_async()

    return "Processed input and started task."


@app.route('/make_dataset_from_global/<square_length>/<center_lat>/<center_lon>')
def run_one_location_from_global_at_center(center_lat,center_lon,square_length=8000):
    # runs model centered at provided coordinates

    center_lat = float(center_lat)
    center_lon = float(center_lon)
    square_length = float(square_length)

    if global_dataset_loaded:
        bl_coords, tr_coords = get_square_corners(center_lat, center_lon, square_length)
        local_dataset = filter_coordinates(global_dataset,
                                           ((bl_coords[0], bl_coords[1]),
                                            (tr_coords[0], tr_coords[1])))


        data_filename = f"latlon_{center_lat:.2f}_{center_lon:.2f}" + "__" + str(int(square_length / 1000)) + '.csv'

        output = None
        try:
            output = io.StringIO()
            local_dataset.to_csv(output, index=False)
            output.seek(0)
        except Exception as e:
            print(f"Couldn't generate local dataset {data_filename} because {e}")

        if output:
            # Send the file as a downloadable response
            return Response(
                output,
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment;filename={data_filename}'
                }
            )
        else:
            return "failed creating the data file"



    else:
        return "Waiting for global_dataset to load, please try again later."



# haven't tested this yet
@app.route('/run_one_model_center/<dataset_filename>/<square_length>/<center_lat>/<center_lon>')
def run_one_model_center(dataset_filename,center_lat,center_lon,square_length=8000):

    params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
              'func_list': ["MSE","COM","EMD"], "data_filename": "",
              "results_type": "default", "coordinates" : []}


    bl_coords, tr_coords = get_square_corners(center_lat,center_lon,square_length)
    local_dataset = filter_coordinates(load_dataset_from_csv(dataset_filename),
                                       ((float(bl_coords[0]), float(bl_coords[1])),
                                        (float(tr_coords[0]), float(tr_coords[1]))))

    try:
        local_dataset.to_csv(dataset_filename, index=False)
    except Exception as e:
        print(f"Couldn't save local dataset {dataset_filename} because {e}")
    cache_datafile(dataset_filename)


    rx_blacklist = [0]

    task1 = celery.signature("tasks.train_one_and_log",
                             args=[{**params,"data_filename": dataset_filename,
                                    "rx_blacklist": rx_blacklist,
                                    "coordinates": [bl_coords, tr_coords]}],
                             options={"queue": "GPU_queue"})
    task1.apply_async()

    return "Processed input and started task."

@app.route('/generate_datasets/<cities_data>/<denylist>')
def generate_datasets(cities_data,denylist):
    global global_dataset
    global global_dataset_loaded

    square_length = 8000 # 8000 meters ~ 5 miles

    # ("Bourzanga, BF", 13.67806, -1.54611)


    # cities_data dict like '4366476': {'geonameid': '4366476', 'name': 'Randallstown', 'latitude': '39.36733', 'longitude': '-76.79525', 'country': 'US', 'population': '32430', 'timezone': 'America/New_York'}
    # cities_data = pickle.load(open('datasets/cities15000_dict_all.pickle', 'rb'))
    cities_data = pickle.load(open(f'datasets/{cities_data}', 'rb'))


    params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
              'func_list': ["PATHLOSS"], "data_filename": "",
              "results_type": "default", "coordinates" : [(47.556372, -122.360229), (47.63509, -122.281609)]}



    if not os.path.exists('/datasets/generated'):
        os.makedirs('/datasets/generated')


    if global_dataset_loaded:
        for city in cities_data:
            lat,lon = float(cities_data[city]['latitude']),float(cities_data[city]['longitude'])
            bl_coords, tr_coords = get_square_corners(lat,lon,square_length)
            local_dataset = filter_coordinates(global_dataset,
                                               ((float(bl_coords[0]), float(bl_coords[1])),
                                                (float(tr_coords[0]), float(tr_coords[1]))))

            data_filename = "datasets/generated/" + cities_data[city]['geonameid'] + "_" +  cities_data[city]["name"] \
            + "__" + cities_data[city]["country"] + str(int(square_length/1000)) + '.csv'

            try:
                local_dataset.to_csv(data_filename, index=False)
            except Exception as e:
                print(f"Couldn't save local dataset {data_filename} because {e}")
            cache_datafile(data_filename)


            if denylist == 'denylist_enabled':
                rx_blacklist = filter_deny_lat(bl_coords[0],tr_coords[0])
            else:
                rx_blacklist = [0]

            task1 = celery.signature("tasks.train_one_and_log",
                                     args=[{**params,"data_filename": data_filename,
                                            "rx_blacklist": rx_blacklist,
                                            "coordinates": [bl_coords, tr_coords]}],
                                     options={"queue": "GPU_queue"})
            task1.apply_async()

        return "Processed cities list and start tasks."



    else:
        return "Waiting for global_dataset to load, please try again later."
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
    # add any directory paths
    # List of directories to search
    directories = ["datasets/helium_SD/", "datasets/external/"]

    try:
        # Initialize a single list to hold all filenames
        all_files = []

        # Iterate through each directory
        for directory_path in directories:
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                # List files in the directory and filter out subdirectories
                files = [
                    f for f in os.listdir(directory_path)
                    if os.path.isfile(os.path.join(directory_path, f))
                ]
                all_files.extend(files)  # Add filenames to the list
            else:
                # Optionally log or skip directories that don't exist
                print(f"Directory not found: {directory_path}")

        # Return a JSON response with all filenames
        return jsonify({"files": all_files})

    except Exception as e:
        # Handle other exceptions
        return jsonify({"error": str(e)}), 500

    #
    # directory_path_external = "external_datasets"
    # directory_path = "datasets/helium_SD/"
    # try:
    #     # Get the list of files in the directory
    #     files = os.listdir(directory_path)
    #     # Filter only files (ignore subdirectories)
    #     file_names = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
    #     return jsonify({"files": file_names})
    # except FileNotFoundError:
    #     # Handle case where the directory does not exist
    #     return jsonify({"error": "Directory not found"}), 404
    # except Exception as e:
    #     # Handle other exceptions
    #     return jsonify({"error": str(e)}), 500
    #
    # return jsonify(res)

@app.route('/selected_data_file', methods=['GET'])
def handle_selected_data_file():
    directories = ["datasets/helium_SD/", "datasets/external/"]
    directory_path = "datasets/helium_SD/"
    filename = request.args.get('filename')
    if filename:
        for directory_path in directories:
            if filename in os.listdir(directory_path):
                res = load_dataset_from_csv(directory_path+filename)

        # Perform some action with the selected filename
        filtered_df = res[['lat1', 'lon1']].dropna().drop_duplicates()

        filtered_df['lat1'] = filtered_df['lat1'].astype(float)
        filtered_df['lon1'] = filtered_df['lon1'].astype(float)

        # Convert the filtered DataFrame to a list of dictionaries
        result = filtered_df.to_dict(orient='records')
        # print(result,flush=True)
        # Return as JSON response
        return jsonify(result)
    return jsonify({"error": "No filename provided"}), 400

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
    global_dataset_filepath = "/datasets/global/all_data.csv" # this maps externally to global dataset file in remote_dev.env
    print("Loading global dataset..",flush=True) # print immediately
    global_dataset_loaded = False # flag updates when global_dataset load completes
    global_dataset = None # placeholder for global variable
    # only uncomment below if needed, because it takes a while to load otherwise
    threading.Thread(target=load_global_dataset, args=(global_dataset_filepath,)).start()
    app.run(host='0.0.0.0',debug=True)
