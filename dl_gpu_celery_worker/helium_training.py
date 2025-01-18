import numpy as np
import torch
import pickle
from localization import DLLocalization
from localization import PhysLocalization
from locconfig import LocConfig
from dataset import RSSLocDataset
from models import CoMLoss, SlicedEarthMoversDistance
import csv
from coordinates import HELIUMSD_LATLON
import os
import math
import code
import matplotlib.pyplot as plt



from attacker import batch_wrapper, get_all_attack_preds_without_grad

# this is intended to be used only for the helium network dataset (data set 9 in dataset.py)
# need to setup this file to read parameters from a json structure that gets passed in..

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

def is_between(number, var1, var2):
    lower_bound = min(var1, var2)
    upper_bound = max(var1, var2)
    #return True
    return lower_bound <= number <= upper_bound

# Returns all unique rx_lat values from the dataset in a list of floats
# assumes ds9 file format where rx_lat (or lat2) would be row[3]
# excludes any rows where TX or RX is outside the boundary of coordinates
# only works for rectangles, won't work with polygons right now
def get_rx_lats(passed_params = {None}):
    rx_lats = set()
    if 'data_filename' in passed_params:
        file_path = passed_params["data_filename"]
    else:
        print(f"ERROR: No data_filename specified in passed_params")
        return []
    if 'coordinates' in passed_params:
        coordinates = passed_params['coordinates']
    else:
        print(f"WARNING: No coordinates specified in passed_params, using defaults")
        coordinates = HELIUMSD_LATLON

    if 'timespan' in passed_params:
        timespan = passed_params['timespan'] # should be tuple like (1718303982,1800000000) or single int 1718303982
    else:
        timespan = 0 # use all samples newer than 0 (i.e. use all)

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header row
        next(csvreader)

        for row in csvreader:
            try:
                lat1 = float(row[1])
                lon1 = float(row[2])
                lat2 = float(row[3])
                lon2 = float(row[4])

                if is_between(lat1, coordinates[0][0],coordinates[1][0]) and \
                    is_between(lon1, coordinates[0][1], coordinates[1][1]) and \
                    is_between(lat2, coordinates[0][0], coordinates[1][0]) and \
                    is_between(lon2, coordinates[0][1], coordinates[1][1]):

                    rx_lat = float(lat2)
                    #print(f"row {row} is between {coordinates}")
                else:
                    #print(f"row {row} is not between {coordinates}")
                    continue

                # # skip if timestamp is outside of timespan; expecting either int or (int,int)
                if isinstance(timespan, int):
                    if int(row[0]) < timespan:
                        continue
                else:
                    if int(row[0]) < timespan[0] or int(row[0]) > timespan[1]:
                        continue

                rx_lats.add(rx_lat)

            except (IndexError, ValueError) as e:
                # Handle the case where the row does not have enough columns
                # or the conversion to float fails
                print(f"Skipping row: {row} due to error: {e}")

    return list(rx_lats)

# returns a list of time_splits from the passed data_file; assumes dataset has already been geo-filtered by coordinates
# does not assume rows are in order and will sort according to row[0] which should be a unix timestamp
# also include the number of rows in each split as the 3rd element in the tuple
def get_time_splits(passed_params = {None}):
    splits = [] # stores timespan segments
    if 'data_filename' in passed_params:
        file_path = passed_params["data_filename"]
    else:
        print(f"ERROR: No data_filename specified in passed_params")
        return []

    if 'timespan' in passed_params:
        timespan = passed_params['timespan'] # should be tuple like (1718303982,1800000000) or single int 1718303982
    else:
        timespan = 0 # use all samples newer than 0 (i.e. use all)

    if 'split_timespan' in passed_params:
        split_timespan = passed_params['split_timespan'] # should be tuple like (1718303982,1800000000) or single int 1718303982
    else:
        split_timespan = 2628288 # one month default

    print(f"timespan and split_timespan are {timespan}, {split_timespan}")

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header row
        next(csvreader)

        sorted_rows = sorted(csvreader, key=lambda row: row[0])

        last_timestamp = 0

        row_count = 0

        for row in sorted_rows:

            current_timestamp = int(row[0])

            # this is for the broader timespan limits
            if isinstance(timespan, int):
                if current_timestamp < timespan:
                    continue
            else:
                if current_timestamp < timespan[0] or current_timestamp > timespan[1]:
                    continue

            row_count += 1
            if current_timestamp > last_timestamp:
                last_timestamp = current_timestamp + split_timespan
                splits.append((current_timestamp, last_timestamp,row_count))
                row_count = 0

    return splits

def plot_splits(rldataset):
    all_tx_vecs = rldataset.data[None].tx_vecs
    keylist = []
    for key in rldataset.data.keys():
        if key == None or "extra" in key: continue
        keylist.append(key)
    print(keylist)
    train_tx_vecs = rldataset.data[keylist[-1]].tx_vecs
    test_tx_vecs = rldataset.data[keylist[-2]].tx_vecs
    val_tx_vecs = rldataset.data[keylist[-3]].tx_vecs

    coordinates = np.array([point[0] for point in all_tx_vecs])
    all_x_coords = coordinates[:, 0]
    all_y_coords = coordinates[:, 1]

    coordinates = np.array([point[0] for point in train_tx_vecs])
    train_x_coords = coordinates[:, 0]
    train_y_coords = coordinates[:, 1]

    coordinates = np.array([point[0] for point in test_tx_vecs])
    test_x_coords = coordinates[:, 0]
    test_y_coords = coordinates[:, 1]

    coordinates = np.array([point[0] for point in val_tx_vecs])
    val_x_coords = coordinates[:, 0]
    val_y_coords = coordinates[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(train_x_coords, train_y_coords, label=keylist[-1], edgecolors='gray', s=100, facecolors='none', alpha=1.0)
    plt.scatter(test_x_coords, test_y_coords, label=keylist[-2], edgecolors='red', s=100, facecolors='red', alpha=1.0)
    plt.scatter(val_x_coords, val_y_coords, label=keylist[-3], edgecolors='black', s=100, facecolors='none',alpha=1.0)


    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Scatter Plot of Coordinates')
    # plt.legend()
    plt.xticks([])  # Remove x-axis tick values
    plt.yticks([])
    # plt.grid()

    filename = f'{rldataset.data_filename.split("\\")[-1].split(".csv")[0]}_{keylist[-2].replace('.','_')}.png'
    print(f"filename is {filename}")

    plt.savefig(filename, dpi=300, bbox_inches='tight')  # High resolution, tight layout
    #
    # plt.show()
    # plt.close()
def main_process(passed_params = {None}):

    # formerly globals **********
    should_train = True
    should_load_model = False
    restart_optimizer = False

    # Specify params
    #max_num_epochs = 200
    include_elevation_map = False  # True

    batch_size = 64 if should_train else 64

    #num_training_repeats = 1

    device = torch.device('cuda')
    #*************************

    # import parameters:
    if 'max_num_epochs' in passed_params:
        max_num_epochs = passed_params['max_num_epochs']
    else:
        max_num_epochs = 201
    if 'num_training_repeats' in passed_params:
        num_training_repeats = passed_params['num_training_repeats']
    else:
        num_training_repeats = 1
    if 'batch_size' in passed_params:
        batch_size = passed_params['batch_size']
    else:
        batch_size = 64
    # if 'data_format' in passed_params:
    #     data_format = passed_params['data_format']
    # else:
    #     data_format = 'helium_ds9'
    if 'data_filename' in passed_params:
        data_filename = passed_params['data_filename'] # else it should default to full helium dataset?
    else:
        data_filename = None
    if 'coordinates' in passed_params:
        coordinates = passed_params['coordinates']
    else:
        coordinates = None
    if 'rx_blacklist' in passed_params:
        rx_blacklist = passed_params['rx_blacklist']
    else:
        rx_blacklist = ["None"]
    if 'func_list' in passed_params:
        func_list = passed_params['func_list']
    else:
        func_list = ["COM","MSE","EMD"] # ["PATHLOSS"] for physics-based non-ML model
    if 'results_type' in passed_params:
        results_type = passed_params['results_type']
    else:
        results_type = "default"
    if 'timespan' in passed_params:
        timespan = passed_params['timespan'] # should be tuple like (1718303982,1800000000) or single int 1718303982
    else:
        timespan = 0 # use all samples newer than 0 (i.e. use all)

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("results"):
        os.makedirs("results")

    global dataset_index
    global meter_scale
    cmd_line_params = []
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--param_selector", type=int, default=-1, help='Index of pair for selecting params')
    # parser.add_argument("--random_ind", type=int, default=-1, help='Random Int for selecting set of params')
    # args = parser.parse_args()

    if results_type == "split_timespan_results":
        results_key = "timespan-" + str(timespan)
        all_results = {results_key : []} # [0] because when we split it up there will only be one element here
        # for default results type it will just be "None" for the key
    else:
        all_results = {rx_blacklist[0] : []} # [0] because when we split it up there will only be one element here
        # for default results type it will just be "None" for the key


    # select the chosen loss_functions:
    loss_funcs = []
    if "COM" in func_list:
        loss_funcs.append(CoMLoss())
    if "MSE" in func_list:
        loss_funcs.append(torch.nn.MSELoss())
    if "EMD" in func_list:
        loss_funcs.append(SlicedEarthMoversDistance(num_projections=100, scaling=0.01, p=1))
    if "PATHLOSS" in func_list:
        loss_funcs = ["PATHLOSS"]


    for random_state in range(0, num_training_repeats):
        for di in [9]:  # ,7,8,9]:
            for split in ['random', 'grid2', 'grid5']:  # , 'grid10']:
                for loss_func in loss_funcs:
                    cmd_line_params.append([di, split, random_state, loss_func])

    # if args.param_selector > -1:
    #     param_list = [cmd_line_params[args.param_selector]]
    # elif args.random_ind > -1:
    #     param_list = [param for param in cmd_line_params if param[3] == args.random_ind]
    # else:
    param_list = cmd_line_params


    for ind, param_set in enumerate(param_list):
        dataset_index, split, random_state, loss_func = param_set
        print(param_set) # *********** other important part **************
        dict_params = {
            "dataset": dataset_index,
            "data_split": split,
            "batch_size": batch_size,
            "random_state": random_state,
            "include_elevation_map": include_elevation_map,
            # "data_format": data_format,
            "data_filename": data_filename,
            "rx_blacklist": rx_blacklist,
            "timespan": timespan,
        }
        params = LocConfig(**dict_params)  # sets up parameters, also some unique defaults depending on the dataset

        loss_label = "com" if isinstance(loss_func, CoMLoss) else "mse" if isinstance(loss_func,
                    torch.nn.MSELoss) else "emd" if isinstance(loss_func, SlicedEarthMoversDistance) else "unknown"
        param_string = f"{params}_{loss_label}"
        modified_param_string = param_string.replace(':', '-')  # added this fix for Win file systems compat.
        print(f"parameter string {modified_param_string}")
        PATH = 'models/%s__model.pt' % modified_param_string
        model_ending = 'train_val.'
        pickle_filename = 'results/augment_%s.pkl' % modified_param_string
        model_filename = PATH.replace('model.', 'model_' + model_ending)

        rldataset = RSSLocDataset(params)  # this loads and processes different datasets ..self.load_data()
        if not rldataset.error_state == False:
            print(f"WARN: helium_training line 256 error_state: {rldataset.error_state}")
            # skip this iteration
            continue

        rldataset.print_dataset_stats()

        # plot the different splits
        plot_splits(rldataset)
        continue

        code.interact(local=locals())

        print("about to run model")

        if "PATHLOSS" in func_list:
            print("executing PATHLOSS model")
            physloc = PhysLocalization(rldataset)
            all_results[rx_blacklist[0]] = {"log10": physloc.test_model(option="log10"),
                                            "log10_per_node": physloc.test_model_per_node_PL(option="log10")}
            # all_results[rx_blacklist[0]] = {"stats": physloc.get_data_distribution_stats()}
            # as expected log10 model is way better, don't bother with linear
                                            #,"rss_dist_ratio" : physloc.test_model(option="rss_dist_ratio"),}
            return all_results

        else: # do regular DL Localization models

            dlloc = DLLocalization(rldataset, loss_object=loss_func)

            if should_train:
                print('Training on dataset', rldataset.train_key, 'for', max_num_epochs, 'epochs')
                print('Training', PATH)
                dlloc.train_model(max_num_epochs, save_model_file=PATH, load_model=should_load_model,
                                  restart_optimizer=restart_optimizer, load_model_file=PATH)
                results = save_results(dlloc, rldataset, model_filename, pickle_filename)
            else:
                results = save_results(dlloc, rldataset, model_filename, pickle_filename)
            for key in results['err']:
                # print("results err mean()")
                #********************* this is the important part ******************************
                print(key, results['err'][key].mean())
                if results_type == "remove_one_results" and "_test" in key:
                    tmp = param_set
                    tmp[-1] = str(tmp[-1]) # convert obj to string so it can be serializable
                    result_row = tmp + [key] + [str(results['err'][key].mean())] # float->str so it's serializable
                    all_results[rx_blacklist[0]].append(result_row)
                    # only the test values, not train/train_val/2test_extra
                if results_type == "split_timespan_results" and "_test" in key:
                    tmp = param_set
                    tmp[-1] = str(tmp[-1]) # convert obj to string so it can be serializable
                    result_row = tmp + [key] + [str(results['err'][key].mean())] # float->str so it's serializable
                    all_results[results_key].append(result_row)
                    # only the test values, not train/train_val/2test_extra

                if results_type == "default": # record all the error types
                    tmp = param_set
                    tmp[-1] = str(tmp[-1])  # convert obj to string so it can be serializable
                    result_row = tmp + [key] + [str(results['err'][key].mean())]  # float->str so it's serializable
                    all_results[rx_blacklist[0]].append(result_row)

    # TODO maybe make this a json.dump string?
    return all_results # return main()


def save_results(dlloc: DLLocalization, rldataset, model_filename, pickle_filename):
    results = get_results(model_filename, dlloc, rldataset)
    with open(pickle_filename, 'wb') as f:
        print('Saving', pickle_filename)
        pickle.dump(results, f)
    return results


def get_results(filename: str, dlloc: DLLocalization, rldataset: RSSLocDataset):
    print('Loading model from %s for testing' % filename)
    dlloc.load_model(filename)
    results = {'err': {}, 'truth': {}, 'preds': {}}
    run_adv_attacks = False

    for key in [rldataset.train_key] + rldataset.test_keys:
        data = rldataset.data[key]
        res = dlloc.test(dataloader=data.ordered_dataloader, save_images=False)
        err = np.array(res['error']).flatten()
        preds = np.array(res['preds'])
        truth = data.ordered_dataloader.dataset.tensors[1].cpu().numpy()
        results['err'][key] = err
        results['truth'][key] = truth * rldataset.params.meter_scale
        results['preds'][key] = preds * rldataset.params.meter_scale
        if key != rldataset.train_key and run_adv_attacks:
            dlloc.set_rss_tensor()
            x_vecs = data.ordered_dataloader.dataset.tensors[0][:256]
            y_vecs = data.ordered_dataloader.dataset.tensors[1][:256]
            img_shape = (rldataset.img_height(), rldataset.img_width())
            rldataset.params.adv_train = True
            adv_pred = batch_wrapper(
                batch_tensors=(x_vecs, y_vecs),
                func=get_all_attack_preds_without_grad,
                batch_size=128,
                return_length=6,
                kwargs={
                    'dlloc': dlloc,
                    'img_shape': img_shape,
                }
            )
            worst_lo, worst_hi, top, drop, hilo, hilotop = adv_pred
            for res, str in [(worst_lo, 'worst_lo'), (worst_hi, 'worst_hi'), (top, 'top'), (drop, 'drop'),
                             (hilo, 'hilo'), (hilotop, 'hilotop')]:
                res = res.cpu().numpy() * rldataset.params.meter_scale
                truth = np.array(y_vecs.cpu())[:, 0, 1:] * meter_scale
                adv_err = np.linalg.norm(truth - res, axis=1)
                results['err'][key + '_' + str] = adv_err
                results['preds'][key + '_' + str] = res

    return results


if __name__ == '__main__':
    #params = {"max_num_epochs": 20, "num_training_repeats": 1, "batch_size": 64,}
    main_process()
