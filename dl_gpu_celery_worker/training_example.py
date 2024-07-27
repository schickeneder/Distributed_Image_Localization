import numpy as np
import torch
import pickle
import argparse
from localization import DLLocalization
from locconfig import LocConfig
from dataset import RSSLocDataset
from models import CoMLoss, SlicedEarthMoversDistance
#from attacker import batch_wrapper, get_all_attack_preds_without_grad
#foo
should_train = True
should_load_model = False
restart_optimizer = False

# Specify params
max_num_epochs = 200
include_elevation_map = False #True

batch_size = 64 if should_train else 64

num_training_repeats = 1

device = torch.device('cuda')
all_results = {}

def main():
    global dataset_index
    global meter_scale
    cmd_line_params = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_selector", type=int, default=-1, help='Index of pair for selecting params')
    parser.add_argument("--random_ind", type=int, default=-1, help='Random Int for selecting set of params')
    args = parser.parse_args()

    for random_state in range(0,num_training_repeats):
        for di in [9]:#,7,8,9]:
            for split in ['random', 'grid2', 'grid5']:#, 'grid10']:
                for loss_func in [ CoMLoss(), torch.nn.MSELoss(), SlicedEarthMoversDistance(num_projections=100, scaling=0.01, p=1)]:
                    cmd_line_params.append([di, split, random_state, loss_func])
    
    if args.param_selector > -1:
        param_list = [cmd_line_params[args.param_selector] ]
    elif args.random_ind > -1:
        param_list = [param for param in cmd_line_params if param[3] == args.random_ind]
    else:
        param_list = cmd_line_params
    
    for ind, param_set in enumerate(param_list):
        dataset_index, split, random_state, loss_func  = param_set
        print(param_set)
        dict_params = {
            "dataset": dataset_index,
            "data_split": split,
            "batch_size":batch_size,
            "random_state":random_state,
            "include_elevation_map":include_elevation_map,
        }
        params = LocConfig(**dict_params) # sets up parameters, also some unique defaults depending on the dataset

        loss_label = 'com' if isinstance(loss_func, CoMLoss) else 'mse' if isinstance(loss_func, torch.nn.MSELoss) else 'emd' if isinstance(loss_func, SlicedEarthMoversDistance) else 'unknown'
        param_string = f"{params}_{loss_label}"
        modified_param_string = param_string.replace(':', '-') # added this fix for Win file systems compat.
        print(f"parameter string {modified_param_string}")
        PATH = 'models/%s__model.pt' % modified_param_string
        model_ending = 'train_val.'
        global all_results
        pickle_filename = 'results/augment_%s.pkl' % modified_param_string
        model_filename = PATH.replace('model.', 'model_' + model_ending)

        rldataset = RSSLocDataset(params) # this loads and processes different datasets ..self.load_data()
        rldataset.print_dataset_stats()

        dlloc = DLLocalization(rldataset, loss_object=loss_func)

        if should_train:
            print('Training on dataset', rldataset.train_key, 'for', max_num_epochs, 'epochs')
            print('Training', PATH)
            dlloc.train_model(max_num_epochs, save_model_file=PATH, load_model=should_load_model, restart_optimizer=restart_optimizer, load_model_file=PATH)
            results = save_results(dlloc, rldataset, model_filename, pickle_filename)
        else:
            results = save_results(dlloc, rldataset, model_filename, pickle_filename)
        for key in results['err']:
            #print("results err mean()")
            print(key, results['err'][key].mean())

            
def save_results(dlloc: DLLocalization, rldataset, model_filename, pickle_filename):
    results = get_results(model_filename, dlloc, rldataset)
    with open(pickle_filename, 'wb') as f:
        print('Saving', pickle_filename)
        pickle.dump(results, f)
    return results

def get_results(filename: str, dlloc: DLLocalization, rldataset: RSSLocDataset):
    print('Loading model from %s for testing' % filename)
    dlloc.load_model(filename)
    results = {'err':{}, 'truth':{}, 'preds':{}}
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
            x_vecs=data.ordered_dataloader.dataset.tensors[0][:256]
            y_vecs=data.ordered_dataloader.dataset.tensors[1][:256]
            img_shape = (rldataset.img_height(), rldataset.img_width())
            rldataset.params.adv_train = True
            adv_pred = batch_wrapper(
                batch_tensors=(x_vecs,y_vecs),
                func=get_all_attack_preds_without_grad,
                batch_size=128,
                return_length=6,
                kwargs={
                    'dlloc': dlloc,
                    'img_shape':img_shape,
                }
            )
            worst_lo, worst_hi, top, drop, hilo, hilotop = adv_pred
            for res, str in [(worst_lo, 'worst_lo'), (worst_hi, 'worst_hi'), (top, 'top'), (drop, 'drop'), (hilo, 'hilo'), (hilotop, 'hilotop')]:
                res = res.cpu().numpy() * rldataset.params.meter_scale
                truth = np.array(y_vecs.cpu())[:,0,1:] * meter_scale
                adv_err = np.linalg.norm(truth - res, axis=1)
                results['err'][key+'_' + str] = adv_err
                results['preds'][key+'_' + str] = res
    return results


if __name__ == '__main__':
    main()
