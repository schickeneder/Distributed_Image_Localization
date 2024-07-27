import coordinates
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import torch
from typing import List
from scipy.spatial.distance import cdist
from cv2 import getAffineTransform, warpAffine
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.model_selection import train_test_split
from IPython import embed

from locconfig import LocConfig

#ds9_path = 'datasets/helium_SD/remove_one_tmp.csv'
# ds9_path = 'datasets/helium_SD/filtered_Seattle_data.csv'
ds9_path = 'datasets/helium_SD/filtered_Seattle_data_all_helium.csv'

data_files = {
        1:['datasets/data1/data1.txt'],
        2:['datasets/data2/data1.txt'],
        3:['datasets/data3/data1.txt', 'datasets/data3/data2.txt'],
        4:['datasets/data4/data1.txt', 'datasets/data4/data2.txt'],
        5:['datasets/data5/data1.txt', 'datasets/data5/data2.txt'],
        6:['datasets/frs_data/separated_data/all_data/no_tx.json', 'datasets/frs_data/separated_data/all_data/single_tx.json', 'datasets/frs_data/separated_data/all_data/two_tx.json'],
        7:['datasets/data_antwerp/lorawan_antwerp_2019_dataset.json.txt'],
        8:['datasets/slc_cbrs_data/slc_prop_measurement/data/data.json'],
        9:[ds9_path], # also need to change it below!
        #9:['datasets/helium_SD/helium_bounded_SD.csv'],
        #9:[r'C:\Users\ps\Helium_Data\all_nz_distances_di9.csv'],
        10:['datasets/helium_SD_SEA/SD_SEA_cleaned.csv'],
}


class RSSLocDataset():
    class Samples():
        def __init__(self, rldataset, rx_vecs=None, tx_vecs=None, filter_boundaries=[], filter_rx=False,
                     tx_metadata=None, no_shift=False):
            """
            rx_vecs: Iterable of Rx rss and locations
            tx_vecs: Iterable of Tx locations
            filter_boundaries: list of np.arrays, coordinates with which to filter rx_vecs and/or tx_vecs. Must be in same coordinate system as rx_vecs and tx_vecs
            """
            assert(rx_vecs is not None and tx_vecs is not None)
            self.made_tensors = False
            self.rldataset = rldataset
            self.origin = np.zeros(2)
            meter_scale = self.rldataset.params.meter_scale
            if not no_shift:
                self.origin = np.array([self.rldataset.min_x, self.rldataset.min_y]) - self.rldataset.buffer*meter_scale
            print(f"max_x {self.rldataset.max_x} min_x {self.rldataset.min_x} diff {self.rldataset.max_x - self.rldataset.min_x}")
            self.rectangle_width = self.rldataset.max_x - self.rldataset.min_x + 2*self.rldataset.buffer*meter_scale
            print(f"max_y {self.rldataset.max_x} min_y {self.rldataset.min_y} diff {self.rldataset.max_y - self.rldataset.min_y}")
            self.rectangle_height = self.rldataset.max_y - self.rldataset.min_y + 2*self.rldataset.buffer*meter_scale
            self.rectangle_width = round(self.rectangle_width / (meter_scale*2)) * meter_scale*2
            self.rectangle_height = round(self.rectangle_height / (meter_scale*2)) * meter_scale*2

            if len(filter_boundaries) > 0:
                for i, filter_boundary in enumerate(filter_boundaries):
                    if not isinstance(filter_boundary, Polygon):
                        filter_boundaries[i] = Polygon(filter_boundary)
                new_tx_vecs = []
                new_rx_vecs = []
                new_tx_metadata = []
                for sample_ind, (tx_vec, rx_vec) in enumerate(zip(tx_vecs, rx_vecs)):
                    if filter_rx:
                        tx_locs, rx_tups = rldataset.filter_bounds(filter_boundaries, tx_coords=tx_vec, rx_coords=rx_vec)
                        if len(tx_locs) == 0:
                            continue
                        new_rx_vecs.append(rx_tups)
                    else:
                        tx_locs = rldataset.filter_bounds(filter_boundaries, tx_coords=tx_vec)
                        if len(tx_locs) == 0:
                            continue
                        new_rx_vecs.append(rx_vec)
                    new_tx_vecs.append(tx_locs)
                    if len(tx_metadata) > 0:
                        metadata = tx_metadata[sample_ind]
                        new_tx_metadata.append(metadata)
                tx_vecs = new_tx_vecs
                rx_vecs = new_rx_vecs
                tx_metadata = new_tx_metadata
            self.tx_vecs = np.array(tx_vecs)
            self.rx_vecs = np.array(rx_vecs, dtype=object)
            self.tx_metadata = np.array(tx_metadata) if tx_metadata is not None else np.array([])

            #print(f"self.tx_vecs {self.tx_vecs}")
            self.max_num_tx = max([len(vec) for vec in self.tx_vecs]) # + [0]) # in some grids we may have no TXers?
            self.max_num_rx = self.rldataset.max_num_rx

        def make_tensors(self):
            self.made_tensors = True

            # Appending the tx and rx distance to yvecs
            if self.tx_vecs[0].shape[-1] < 3:
                if len(self.tx_vecs.shape) == 3:
                    tmp_shape = list(self.tx_vecs.shape)
                    tmp_shape[-1] = tmp_shape[-1] + 2
                    tmp_tx_vecs = np.zeros(tmp_shape)
                else:
                    tmp_tx_vecs = self.tx_vecs
                for i, (rx_vec, tx_vec) in enumerate(zip(self.rx_vecs, self.tx_vecs)):
                    if len(tx_vec) == 0: continue
                    tx_distances = []
                    rx_distances = []
                    for tx in tx_vec:
                        min_tx_distance = np.ma.masked_equal(np.linalg.norm( (tx[0:2] - tx_vec[:,0:2]).astype(float), axis=1), 0, copy=False).min()
                        min_rx_distance = np.linalg.norm( (tx[0:2] - rx_vec[:,1:3]).astype(float), axis=1).min()
                        tx_distances.append(min_tx_distance)
                        rx_distances.append(min_rx_distance)
                    tx_distances = np.array(tx_distances).reshape(-1,1)
                    rx_distances = np.array(rx_distances).reshape(-1,1)
                    tmp_tx_vecs[i] = np.hstack((self.tx_vecs[i], tx_distances, rx_distances))
                self.tx_vecs = tmp_tx_vecs
            else:
                for i, (rx_vec, tx_vec) in enumerate(zip(self.rx_vecs, self.tx_vecs)):
                    if len(tx_vec) == 0: continue
                    rx_distances = []
                    for tx in tx_vec:
                        min_rx_distance = np.linalg.norm((tx[0:2] - rx_vec[:,1:3]).astype(float), axis=1).min()
                        rx_distances.append(min_rx_distance)
                    rx_distances = np.array(rx_distances)
                    self.tx_vecs[i][:,-1] = rx_distances

            num_params = 2
            rx_vecs_arr = np.zeros((len(self.rx_vecs),self.max_num_rx+(10 if self.rldataset.params.adv_train else 0), num_params+3))
            for i, rx_vec in enumerate(self.rx_vecs):
                if len(rx_vec):
                    rx_vec = rx_vec / np.array([1,self.rldataset.params.meter_scale, self.rldataset.params.meter_scale,1,1])
                    rx_vecs_arr[i,rx_vec[:,3].astype(int)] = rx_vec
                    self.rx_vecs[i][:,1:3] -= self.origin
            rx_vecs_arr[:,:,1:3] -= self.origin/self.rldataset.params.meter_scale # Adjust coordinates to the bottom left corner of rectangle as origin
            rx_vecs_arr[rx_vecs_arr < 0] = 0
            rx_vecs_tensor = torch.Tensor(rx_vecs_arr).to(self.rldataset.params.device)

            tx_vecs_arr = np.zeros((len(self.tx_vecs),self.max_num_tx, num_params+1))
            for i, tx_vec in enumerate(self.tx_vecs):
                if len(tx_vec):
                    tx_vecs_arr[i,:len(tx_vec),0] = 1
                    tx_vecs_arr[i,:len(tx_vec),1:num_params+1] = (np.array(tx_vec)[:,:num_params] - self.origin) / np.array([self.rldataset.params.meter_scale, self.rldataset.params.meter_scale])
                    self.tx_vecs[i][:,0:2] -= self.origin
            tx_vecs_tensor = torch.Tensor(tx_vecs_arr).to(self.rldataset.params.device)
            dataset = torch.utils.data.TensorDataset(rx_vecs_tensor, tx_vecs_tensor)
            pin_memory = self.rldataset.params.device != torch.device('cuda') 
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.rldataset.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.rldataset.params.batch_size, shuffle=False, pin_memory=pin_memory)


    def __init__(self, params: LocConfig, sensors_to_remove: List[int]=[]):
        """
        params: a LocConfig object, see config.py for parameters
        sensors_to_remove: List[int] input indicies to remove from training data, for experiments on adding new devices.
        """
        self.params = params
        self.data = {}
        self.buffer = 1
        self.data_files = data_files[self.params.dataset_index]
        self.load_data()
        self.elevation_tensors = None
        self.building_tensors = None
        self.make_datasets(
            split=self.params.data_split,
            make_val=self.params.make_val,
            should_augment=self.params.should_augment,
            convert_all_inputs=True,
            sensors_to_remove=sensors_to_remove
        )


    def make_elevation_tensors(self, meter_scale=None):
        meter_scale = self.params.meter_scale if meter_scale is None else meter_scale
        corners = self.corners.copy()
        if len(corners) == 4:
            height = int(np.linalg.norm(corners[0] - corners[1]).round())
            width = int(np.linalg.norm(corners[1] - corners[2]).round())
        else:
            height = corners[1,1] - corners[0,1]
            width = corners[1,0] - corners[0,0]
            corners = np.array([corners[0], [corners[0,0], corners[1,1]], corners[1] ])

        transform = np.array(self.elevation_map.transform).reshape(3,3)
        transform[:2,2] -= self.elevation_map.origin
        building_tensor = None
        inv_transform = np.linalg.inv(transform)
        if len(corners) == 4:
            padded_coordinates = np.hstack((corners, np.ones((4,1)) )).T
            img_coords = (inv_transform @ padded_coordinates)[:2].T

            img_to_rect_transform = getAffineTransform(img_coords[:3].astype(np.float32), np.float32([[0,0],[0,height], [width,height]]))
            warped_img = warpAffine(self.elevation_map.read(1), img_to_rect_transform, (width, height) )
            if self.building_map is not None:
                building_img = warpAffine(self.building_map.read(1), img_to_rect_transform, (width, height) )
            img = warped_img
            downsample_rate = round(meter_scale) #Rounding since resolution is a messy float
        else:
            a,b = corners.min(axis=0), corners.max(axis=0)
            a_ind = (inv_transform @ np.array([a[0], a[1], 1]) ).round().astype(int)
            b_ind = (inv_transform @ np.array([b[0], b[1], 1]) ).round().astype(int)
            sub_img = self.elevation_map.read(1)[b_ind[1]:a_ind[1]+1, a_ind[0]:b_ind[0]+1]
            #sub_img = elevation_map.read(1)[a_ind[0]:b_ind[0]+1, b_ind[1]:a_ind[1]+1]
            sub_img = np.flipud(sub_img)
            if self.building_map is not None:
                building_img = self.building_map.read(1)[b_ind[1]:a_ind[1]+1, a_ind[0]:b_ind[0]+1]
                building_img = np.flipud(building_img)
            img = sub_img
            downsample_rate = round(meter_scale / self.elevation_map.res[0]) #Rounding since resolution is a messy float
            # Could also use avg_pool2d, not sure that it makes much of a difference.
        elevation_tensor = torch.nn.functional.max_pool2d(torch.tensor(img.copy()).unsqueeze(0), downsample_rate) 
        elevation_tensor = (elevation_tensor - elevation_tensor.min()) / 300
        if self.building_map is not None:
            building_tensor = torch.nn.functional.max_pool2d(torch.tensor(building_img.copy()).unsqueeze(0), downsample_rate) 
        self.elevation_tensors = elevation_tensor
        self.building_tensors = building_tensor


    def separate_dataset(self, separation_method, excluded_metadata={'stationary':'stationary', 'transport':'inside'} ,data_key_prefix='', grid_size=5, train_split=0.8, source_key=None, keys=['train', 'test'], random_state=None):
        inds = self.filter_inds_by_metadata(excluded_metadata, source_key=source_key)
        if separation_method == 'grid':
            (train_inds, train_grid_inds), (test_inds, test_grid_inds) = self._get_random_grid(inds, grid_size=grid_size, train_split=train_split, randomly_add_mixed=False, source_key=source_key, random_state=random_state)
            separation_method = separation_method + str(grid_size)
        elif separation_method == 'walking':
            train_inds, test_inds = self._metadata_filter(inds, search_key='transport', search_value='walking', source_key=source_key)
        elif separation_method == 'driving':
            train_inds, test_inds = self._metadata_filter(inds, search_key='transport', search_value='driving', source_key=source_key)
        elif separation_method == 'station':
            train_inds, test_inds = self._metadata_filter(inds, search_key='stationary', source_key=source_key)
        elif separation_method == 'mobile':
            test_inds, train_inds = self._metadata_filter(inds, search_key='stationary', source_key=source_key)
        else: 
            raise NotImplementedError
        rx_vecs = self.data[source_key].rx_vecs
        tx_vecs = self.data[source_key].tx_vecs

        if len(data_key_prefix) == 0:
            train_key = separation_method + '_' + keys[0] 
            test_key = separation_method + '_' + keys[1]
        else:
            train_key = data_key_prefix+ separation_method + '_' + keys[0]
            test_key = data_key_prefix+ separation_method + '_' + keys[1]
        x_vecs_train, x_vecs_test = rx_vecs[train_inds], rx_vecs[test_inds]
        label_train, label_test = tx_vecs[train_inds], tx_vecs[test_inds]
        metadata = self.data[source_key].tx_metadata
        if len(metadata) > 0:
            metadata_train, metadata_test = metadata[train_inds], metadata[test_inds]
            self.data[train_key] = self.Samples(self, x_vecs_train, label_train, tx_metadata=metadata_train, no_shift=True)
            self.data[test_key] = self.Samples(self, x_vecs_test, label_test, tx_metadata=metadata_test, no_shift=True)
        else:
            self.data[train_key] = self.Samples(self, x_vecs_train, label_train, no_shift=True)
            self.data[test_key] = self.Samples(self, x_vecs_test, label_test, no_shift=True)
        if 'grid' in separation_method:
            self.data[train_key].grid_inds = train_grid_inds
            self.data[test_key].grid_inds = test_grid_inds
        return train_key, test_key


    def filter_inds_by_metadata(self, excluded_metadata={'stationary':'stationary', 'transport':'inside'}, source_key=None):
        inds = np.arange(len(self.data[source_key].tx_vecs))
        for key in excluded_metadata:
            value = excluded_metadata[key]
            _, inds = self._metadata_filter(inds, search_key=key, search_value=value)
        return inds


    def _metadata_filter(self, inds, search_key, source_key=None, search_value=None):
        """
        Split the dataset by metadata terms:
        stationary: an int index, where samples with the same index are taken at the same location
        transport: 'driving' or 'walking'
        radio: 'Audiovox', 'TXA', or 'TXB'
        power: 0.5 or 1 (Audiovox or Baofeng)
        """
        inds_with_term = []
        inds_without_term = []
        tx_metadata = self.data[source_key].tx_metadata
        if len(tx_metadata) == 0 or self.params.dataset_index != 6:
            return inds_with_term, inds
        for ind in inds:
            if search_key == 'stationary':
                if 'stationary' in tx_metadata[ind][0]:
                    ### Should we have a function here to combine stationary samples into one huge sample?
                    inds_with_term.append(ind)
                else:
                    inds_without_term.append(ind)
            else:
                if tx_metadata[ind][0][search_key] == search_value:
                    inds_with_term.append(ind)
                else:
                    inds_without_term.append(ind)
        inds_with_term, inds_without_term = np.array(inds_with_term, dtype=int), np.array(inds_without_term, dtype=int)
        return inds_with_term, inds_without_term
    
    def _get_random_grid(self, inds, grid_size=5, train_split=0.5, randomly_add_mixed=True, x_grid_size=None, y_grid_size=None, source_key=None, random_state=None):
        source = self.data[source_key]
        if x_grid_size is None:
            x_grid_size = grid_size
        if y_grid_size is None:
            y_grid_size = grid_size
        min_bounds = source.tx_vecs.min(axis=0)[0,:2] - 1
        max_bounds = source.tx_vecs.max(axis=0)[0,:2] + 1
        x_bins = np.linspace(min_bounds[0], max_bounds[0], x_grid_size+1)
        y_bins = np.linspace(min_bounds[1], max_bounds[1], y_grid_size+1)
        self.x_grid_lines = x_bins
        self.y_grid_lines = y_bins
        if random_state is None:
            random_state = self.params.random_state
        np.random.seed(random_state)
        choices = list(range(x_grid_size*y_grid_size))
        train_grid_flat = list(range(x_grid_size*y_grid_size))
        if hasattr(source, 'grid_inds'):
            choices = copy.deepcopy(source.grid_inds)
            train_grid_flat = copy.deepcopy(source.grid_inds)
        train_size = round(len(choices) * train_split)
        test_size = len(choices) - train_size
        test_grid_flat = []
        backup_inds = []
        while len(test_grid_flat) < test_size:
            choice = np.random.choice(choices)
            choices.remove(choice)
            train_grid_flat.remove(choice)
            for to_remove in [choice+1, choice-1, choice+x_grid_size, choice-x_grid_size]:
                try:
                    choices.remove(to_remove)
                    backup_inds.append(to_remove)
                except:
                    pass
            test_grid_flat.append( choice )
            if len(choices) == 0:
                ind_need = test_size - len(test_grid_flat)
                new_inds = np.random.choice(backup_inds, size=ind_need).tolist()
                test_grid_flat += new_inds
                for ind in new_inds:
                    train_grid_flat.remove(ind)
                break

        train_inds = []
        test_inds = []
        for ind in inds:
            if len(source.tx_vecs[ind]) == 0: # If there are no transmitters (0 Tx)
                if np.random.random() <= train_split:
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
                continue
            tx_vec = source.tx_vecs[ind] # - source.origin
            x_inds = np.digitize(tx_vec[:,0], x_bins) - 1
            y_inds = np.digitize(tx_vec[:,1], y_bins) - 1
            grid_inds = np.ravel_multi_index((x_inds, y_inds), (x_grid_size, y_grid_size))
            ## I'm not sure if this is still working, with the two separate x and y grid divisions.
            if all(x in train_grid_flat for x in grid_inds):
                train_inds.append(ind)
            elif all(x not in train_grid_flat for x in grid_inds):
                test_inds.append(ind)
            elif randomly_add_mixed: # If transmitters occur in both train and test grids, randomly assign them to either side
                if np.random.random() >= train_split:
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
            else: # If we don't add mixed transmitters randomly, just add them to test
                test_inds.append(ind)
        return (train_inds, train_grid_flat), (test_inds, test_grid_flat)
        

    def get_data_within_radius(self, point, radius, data_key=None, new_key=None, out_key=None, percent_filtered=1):
        assert (new_key is None and out_key is None) or (new_key is not None and out_key is not None)
        if new_key is None:
            if percent_filtered < 1:
                new_key = '%s_inradius%i_%.1f' % (str(data_key), radius, percent_filtered)
                out_key = '%s_outradius%i_%.1f' % (str(data_key), radius, percent_filtered)
            else:
                new_key = '%s_inradius%i' % (str(data_key), radius)
                out_key = '%s_outradius%i' % (str(data_key), radius)
        assert data_key in self.data
        source = self.data[data_key]
        inds_within_radius = []
        inds_without_radius = []
        for ind, tx_vec in enumerate(source.tx_vecs):
            for tx in tx_vec:
                if np.linalg.norm(tx[:2] - point) <= radius:
                    inds_within_radius.append(ind)
                    break
                else:
                    inds_without_radius.append(ind)
                    break
        if len(source.tx_metadata):
            self.data[new_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_within_radius], tx_vecs=source.tx_vecs[inds_within_radius], tx_metadata=source.tx_metadata[inds_within_radius], no_shift=True)
            self.data[out_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_without_radius], tx_vecs=source.tx_vecs[inds_without_radius], tx_metadata=source.tx_metadata[inds_without_radius], no_shift=True)
        else:
            self.data[new_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_within_radius], tx_vecs=source.tx_vecs[inds_within_radius], no_shift=True)
            self.data[out_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_without_radius], tx_vecs=source.tx_vecs[inds_without_radius], no_shift=True)
        return new_key, out_key


    def make_datasets(self, split=None, make_val=True, eval_train=False, eval_special=False, train_size=None, should_augment=False, synthetic_only=False, convert_all_inputs=False, sensors_to_remove=[]):
        params = self.params
        if split==None:
            split = params.data_split
        train_key=None
        test_keys = []
        special_keys = []
        data_key_prefix = '%.1ftestsize' % params.test_size
        if split == 'random' or split == 'random_limited':
            if data_key_prefix + '_train' in self.data.keys():
                return '0.2testsize_train', '0.2testsize_test'
            if self.params.dataset_index in [6,8]:
                if 'campus' not in self.data.keys():
                    self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
                train_key, test_key = self.separate_random_data(test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, data_source_key='campus', random_state=0) # We fix the random state so we always have the same test set, but different validation sets.
            else:
                train_key, test_key = self.separate_random_data(test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, random_state=0) # We fix the random state so we always have the same test set, but different validation sets.
            if make_val:
                train_key, train_val_key = self.separate_random_data(test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, data_source_key=train_key, ending_keys=['train', 'train_val'])
                test_keys = [test_key, train_val_key]
            else:
                test_keys = [test_key]
        elif 'grid' in split:
            train_random, test_random = self.make_datasets(make_val=False, split='random')
            random_state = 0 
            grid_size = int(split.split('grid')[1])
            if grid_size == 2 and self.params.dataset_index == 8 and self.params.use_alt_for_ds8_grid2:
                random_state = 1
            train_key, test_key = self.separate_dataset('grid', grid_size=grid_size, data_key_prefix=data_key_prefix, train_split=params.training_size, source_key=train_random, random_state=random_state)
            train_val_key, test_extra_key = self.separate_dataset('grid', grid_size=grid_size, data_key_prefix=data_key_prefix, train_split=params.training_size, source_key=test_random, random_state=random_state, keys=['train_val', '2test_extra'])
            test_keys = [test_key, train_val_key]
            if len(self.data[test_key].rx_vecs) < 100:
                test_keys.append(test_extra_key)
        elif 'april' in split or 'july' in split or 'nov' in split:
            if 'campus' not in self.data.keys():
                self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
            source = self.data['campus']
            april_metadata = np.array(['2022-04' in meta[0]['time'] for meta in source.tx_metadata])
            july_metadata = np.array(['2022-07' in meta[0]['time'] for meta in source.tx_metadata])
            nov_metadata = np.array(['2022-11' in meta[0]['time'] for meta in source.tx_metadata])
            april_inds = np.where(april_metadata)
            july_inds = np.where(july_metadata)
            nov_inds = np.where(nov_metadata)
            combined_inds = np.where(april_metadata + july_metadata)
            self.data['april'] = self.Samples(self, rx_vecs=source.rx_vecs[april_inds], tx_vecs=source.tx_vecs[april_inds], tx_metadata=source.tx_metadata[april_inds])
            self.data['july'] = self.Samples(self, rx_vecs=source.rx_vecs[july_inds], tx_vecs=source.tx_vecs[july_inds], tx_metadata=source.tx_metadata[july_inds])
            self.data['nov'] = self.Samples(self, rx_vecs=source.rx_vecs[nov_inds], tx_vecs=source.tx_vecs[nov_inds], tx_metadata=source.tx_metadata[nov_inds])
            if 'combined' in split:
                self.data['combined'] = self.Samples(self, rx_vecs=source.rx_vecs[combined_inds], tx_vecs=source.tx_vecs[combined_inds], tx_metadata=source.tx_metadata[combined_inds])
            if 'april' in split[:5]:
                train_key = 'april'
                test_keys = ['nov'] if 'nov' in split else ['july']
            elif 'july' in split[:4]:
                train_key = 'july'
                test_keys = ['nov'] if 'nov' in split else ['april']
            elif 'nov' in split[:3]:
                train_key = 'nov'
                test_keys = ['april'] if 'april' in split else ['july']

            if 'combined' in split:
                train_key = 'combined'
                test_keys = ['nov', 'april', 'july']

            prefix = train_key
            if 'selftest' in split:
                _, ood_test_key = self.separate_random_data('april', random_state=0, data_key_prefix='april')
                _, ood_test_key = self.separate_random_data(test_keys[0], random_state=0, data_key_prefix=test_keys[0])
                train_key, id_test_key = self.separate_random_data(train_key, random_state=0, data_key_prefix=train_key)
                test_keys = [id_test_key, ood_test_key]
            if make_val:
                train_key, train_val_key = self.separate_random_data(train_key, random_state=params.random_state, data_key_prefix=prefix, ending_keys=['train', 'train_val'])
                test_keys = test_keys + [train_val_key]
            test_key = test_keys[0]
        elif 'driving' in split:
            self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
            split_keys = ['driving', 'non-driving']
            driving_key, non_driving_key = self.separate_dataset('driving', keys=split_keys, source_key='campus')
            if split == 'driving':
                train_key, test_key = driving_key, non_driving_key
            else:
                train_key, test_key = non_driving_key, driving_key
            if make_val:
                train_key, train_val_key = self.separate_random_data(train_key, random_state=params.random_state, ending_keys=['train', 'train_val'])
                test_keys = [test_key, train_val_key]
            else:
                test_keys = [test_key]
        elif 'biking' in split:
            self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
            split_keys = ['driving', 'non-driving']
            driving_key, non_driving_key = self.separate_dataset('driving', keys=split_keys, source_key='campus')
            walking_key, biking_key = self.separate_dataset('walking', keys=['walking', 'biking'], source_key=non_driving_key)
            train_key, test_key = biking_key, walking_key
            test_keys = [walking_key, driving_key]
            if make_val:
                train_key, train_val_key = self.separate_random_data(train_key, random_state=params.random_state, ending_keys=['train', 'train_val'])
                test_keys = test_keys + [train_val_key]
        elif 'radius' in split:
            keys = self.make_datasets(make_val=make_val, split='random')
            radius = int(split.split('radius')[1])
            center_point = self.get_center_point()
            train_key, _ = self.get_data_within_radius(center_point, radius, data_key=self.train_key)
            test_keys, train_val_keys = [self.get_data_within_radius(center_point, radius, data_key=key) for key in self.test_keys]
            test_key = test_keys[1]
            test_keys = [test_keys[1], train_val_keys[0]]


        if 'off_campus' == split or eval_special:
            if 'campus' not in self.data.keys():
                self.make_filtered_sample_source([coordinates.campus_polygon], 'campus')
            if '0.2testsize_train' not in self.data.keys():
                self.make_datasets(make_val=False, split='random')
            self.data['off_campus'] = self.make_missing_samples('campus')
            train_key = '0.2testsize_train'
            if eval_special:
                special_keys.append('off_campus')
            else:
                test_key = 'off_campus'
        if 'indoor' == split or eval_special:
            if '0.2testsize_train' not in self.data.keys():
                self.make_datasets(make_val=False, split='random')
            source = self.data[None]
            metadata = np.array([meta[0]['transport'] for meta in source.tx_metadata])
            inds = np.where(metadata == 'inside')
            self.data['indoor'] = self.Samples(self, rx_vecs=source.rx_vecs[inds], tx_vecs=source.tx_vecs[inds], tx_metadata=source.tx_metadata[inds])
            train_key = '0.2testsize_train'
            if eval_special:
                special_keys.append('indoor')
            else:
                test_key = 'indoor'
        if split == '2tx' or eval_special:
            params2 = copy.deepcopy(params)
            params2.make_val = False
            params2.one_tx = False
            params2.data_split = 'random'
            rldataset2 = RSSLocDataset(params2)
            source = rldataset2.data[None]
            tx_lengths = np.array([len(tx_vec) for tx_vec in source.tx_vecs])
            inds = np.where(tx_lengths == 2)
            tx_vecs = np.array(list(source.tx_vecs[inds]))
            self.data['2tx'] = self.Samples(self, rx_vecs=source.rx_vecs[inds], tx_vecs=tx_vecs, tx_metadata=source.tx_metadata[inds])
            if '0.2testsize_train' not in self.data.keys():
                self.make_datasets(make_val=False, split='random')
            train_key = '0.2testsize_train'
            if eval_special:
                special_keys.append('2tx')
            else:
                test_key = '2tx'

        if eval_train:
            test_keys.append(train_key)

        if len(test_keys) > 0:
            self.set_default_keys(train_key=train_key, test_keys=test_keys + special_keys)
        else:
            self.set_default_keys(train_key=train_key, test_key=test_key)

        if should_augment and params.augmentation is not None:
            # Must indicate if sensors should be removed BEFORE training the augmentor object
            self.sensors_to_remove = sensors_to_remove
            self.add_synthetic_training_data(synthetic_only=synthetic_only, convert_all_inputs=convert_all_inputs)
            #self.nonlearning_localization()
        # Remove sensors from train data for experiments on adding new devices at inference time
        if sensors_to_remove:
            for sensor in sensors_to_remove:
                self.mute_inputs_in_data_key(train_key, sensor)

        return train_key, test_key

    
    def mute_inputs_in_data_key(self, key, sensor_id, mute_synthetic=False):
        dataset = self.data[key].ordered_dataloader.dataset
        if mute_synthetic:
            dataset.tensors[0][:,sensor_id] = 0
        else:
            mute_length = len(self.data[key].rx_vecs)
            dataset.tensors[0][:mute_length,sensor_id] = 0


    def nonlearning_localization(self):
        rx_data_tr = self.data[self.train_key].ordered_dataloader.dataset.tensors[0]
        tx_locs_tr = self.data[self.train_key].ordered_dataloader.dataset.tensors[1][:,:,1:]
        rx_data_tv = self.data[self.test_keys[1]].ordered_dataloader.dataset.tensors[0]
        tx_locs_tv = self.data[self.test_keys[1]].ordered_dataloader.dataset.tensors[1][:,:,1:]
        rx_data_te = self.data[self.test_keys[0]].ordered_dataloader.dataset.tensors[0]
        tx_locs_te = self.data[self.test_keys[0]].ordered_dataloader.dataset.tensors[1][:,:,1:]
        keys = self.prop_keys
        eps = 0.05
        tr_err = np.zeros(300) #len(tx_locs_tr))
        tv_err = np.zeros(300) #len(tx_locs_tv))
        te_err = np.zeros(300) #len(tx_locs_te))
        tr_err1 = [np.zeros(300) for i in range(5)] #len(tx_locs_tr))
        tv_err1 = [np.zeros(300) for i in range(5)] #len(tx_locs_tr))
        te_err1 = [np.zeros(300) for i in range(5)] #len(tx_locs_tr))
        #tr_err1 = np.zeros(300) #len(tx_locs_tr))
        #tv_err1 = np.zeros(300) #len(tx_locs_tv))
        #te_err1 = np.zeros(300) #len(tx_locs_te))
        for rx_set, tx_set, err_set, err_set1 in zip(
            [rx_data_tr, rx_data_tv, rx_data_te],
            [tx_locs_tr, tx_locs_tv, tx_locs_te],
            [tr_err, tv_err, te_err],
            [tr_err1, tv_err1, te_err1],
        ):
            for i in range(300): #len(tx_set)):
                input = rx_set[i,:,0]
                inp = input[keys][input[keys].cpu().numpy() > 0].cpu().numpy()
                tx = tx_set[i,0].cpu().numpy()
                inp_maps = self.prop_maps[input[keys].cpu() > 0].cpu().numpy()
                inp_maps[np.isnan(inp_maps)] = 0
                pred_maps = (inp_maps >= inp[:,None,None] - eps) * (inp_maps <= inp[:,None,None] + eps)
                pred_map = pred_maps.sum(axis=0)
                err = np.linalg.norm(np.fliplr(np.array(np.where(pred_map == pred_map.max())).T) - tx, axis=1).min() * self.params.meter_scale
                err_set[i] = err
                pred_maps = (1 - abs(inp_maps - inp[:,None,None])) * inp[:,None,None]
                pred_map = pred_maps.sum(axis=0)
                err = np.linalg.norm(np.fliplr(np.array(np.where(pred_map == pred_map.max())).T) - tx, axis=1).min() * self.params.meter_scale
                err_set1[1][i] = err
        print(tr_err.mean(), tv_err.mean(), te_err.mean())
        print(tr_err1[1].mean(), tv_err1[1].mean(), te_err1[1].mean())
        #print(np.median(tr_err), np.median(tv_err), np.median(te_err), np.median(tr_err1), np.median(tv_err1), np.median(te_err1))
        embed()
        exit


    def add_synthetic_training_data(self, method=None, synthetic_only=False, convert_all_inputs=True, synthetic_sample_distance_in_pixels=1):
        from synthetic_augmentations import SyntheticAugmentor
        method = self.params.augmentation if method is None else method
        if method not in ['linear', 'nearest', 'rbf', 'tirem', 'tirem_nn', 'celf']:
            return
        augmentor = SyntheticAugmentor(method=method, rldataset=self)
        augmentor.train()
        prop_results = augmentor.get_results()
        prop_maps, keys, rx_locs, rx_types, rx_test_preds, rx_test_rss, tx_test_coords = prop_results
        keys = [key[0] for key in keys]

        if self.params.dataset_index == 6:
            rm13 = keys.index(13)
            keys.pop(rm13)
            rx_locs.pop(rm13)
            rx_types.pop(rm13)
            rm13_mask = np.ones(len(prop_maps), dtype=bool)
            rm13_mask[rm13] = False
            prop_maps = prop_maps[rm13_mask]

        train_data = self.data[self.train_key]
        tx_data = train_data.ordered_dataloader.dataset.tensors[1]
        tx_locs = tx_data[:,:,1:]
        rx_data = train_data.ordered_dataloader.dataset.tensors[0]
        self.prop_maps = torch.tensor(prop_maps, dtype=torch.float32, device=self.params.device)
        grid_bounds = [[1,self.img_width()-1], [1,self.img_height()-1]]
        if self.params.dataset_index == 6:
            grid_bounds = [[3,70], [15,self.img_height()]]
        grid = np.array(np.meshgrid(np.arange(grid_bounds[0][0],grid_bounds[0][1]), np.arange(grid_bounds[1][0],grid_bounds[1][1]))).reshape(2,-1).T 
        if synthetic_only:
            new_tx_locs = grid
        else:
            nearest_tx = cdist(tx_locs.squeeze().cpu(), grid+0.5).min(axis=0)
            new_tx_locs = grid[nearest_tx > synthetic_sample_distance_in_pixels]
        new_rss = self.prop_maps[:, new_tx_locs[:,1], new_tx_locs[:,0]].T
        new_rss = new_rss.maximum(torch.zeros((1,), device=self.params.device))
        self.prop_keys = keys
        new_tx = torch.zeros((len(new_tx_locs), tx_locs.shape[1], 3), device=self.params.device)
        new_rx = torch.zeros((len(new_tx_locs), rx_data.shape[1], rx_data.shape[2]), device=self.params.device)
        new_tx[:,:,0] = 1
        new_tx[:,:,1:3] = torch.tensor(new_tx_locs, dtype=torch.float32, device=self.params.device).unsqueeze(1)
        new_rx[:,keys,0] = new_rss
        new_rx[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
        new_rx[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
        new_rx[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
        pin_memory = self.params.device != torch.device('cuda') 

        if convert_all_inputs:
            key = self.train_key
            #for key in [self.train_key] + self.test_keys[1:]:
            self.data[key+'_synthetic'] = copy.copy(self.data[key])
            tx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[1]
            tx_locs = tx_data[:,0,1:].round().int()
            rx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[0]
            pred_rss = self.prop_maps[:, tx_locs[:,1], tx_locs[:,0]].T
            pred_rss = pred_rss.maximum(torch.zeros((1,), device=self.params.device))
            zero_rss_inds = rx_data[:,keys,1] == 0
            rx_data[:,keys,0][zero_rss_inds] = pred_rss[zero_rss_inds]
            rx_data[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
            rx_data[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
            rx_data[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
            if key == self.train_key:
                train_rx, train_tx = rx_data, tx_data
            #self.test_keys = [self.test_keys[0]] + [key+'_synthetic' for key in self.test_keys]
        else:
            train_rx, train_tx = rx_data, tx_data

        if synthetic_only:
            rx, tx = new_rx, new_tx
        else:
            rx, tx = torch.cat((train_rx, new_rx)), torch.cat((train_tx, new_tx))
        dataset = torch.utils.data.TensorDataset(rx, tx)
        key = self.train_key
        self.data[key+'_synthetic'] = copy.copy(self.data[key])
        self.data[key+'_synthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
        self.data[key+'_synthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
        self.train_key = self.train_key + '_synthetic'
        if method == 'fusion':
            rx_data = self.data[self.train_key].ordered_dataloader.dataset.tensors[0]
            tx_data = self.data[self.train_key].ordered_dataloader.dataset.tensors[1]
            coords = tx_data[:,0,1:].round().int().cpu()
            self.synthetic_rss = torch.tensor(self.all_prop_maps[:,:,coords[-len(new_rx):,1], coords[-len(new_rx):,0]]).to(self.params.device)
    

    def get_center_point(self):
        if self.params.dataset_index < 6 or self.params.dataset_index == 8:
            center_point = np.array([(self.max_x - self.min_x)/2, (self.max_y - self.min_y)/2])
        elif self.params.dataset_index == 6:
            center_point = np.array([1100,1200])
        elif self.params.dataset_index == 7:
            center_point = np.array([2500,3000])
        return center_point
            

    def filter_bounds(self, boundary, tx_coords=None, rx_coords=None):
        """
        boundary is either:
            a (2,2) numpy array with the bottom left and top right corners of a bounding box
            a shapely Polygon
            a list of shapely Polygons
        """
        if tx_coords is not None:
            if isinstance(boundary, Polygon) or isinstance(boundary[0], Polygon):
                good_inds = []
                for ind, coord in enumerate(tx_coords):
                    point = Point(coord[0], coord[1])
                    if (isinstance(boundary, Polygon) and boundary.contains(point)) or any(bound.contains(point) for bound in boundary):
                        good_inds.append(ind)
                tx_coords = tx_coords[np.array(good_inds).astype(int)]
            else:
                assert(len(boundary) == 2)
                lat_min, lon_min = boundary.min(axis=0)
                lat_max, lon_max = boundary.max(axis=0)
                lat_violation = sum(tx_coords[:,0] < lat_min) + sum(tx_coords[:,0] > lat_max)
                lon_violation = sum(tx_coords[:,1] < lon_min) + sum(tx_coords[:,1] > lon_max)
                if lat_violation or lon_violation:
                    tx_coords = np.empty((0,2))
            if rx_coords is None:
                return tx_coords
        if rx_coords is not None:
            if isinstance(boundary, Polygon):
                good_inds = []
                for ind, coord in enumerate(rx_coords):
                    point = Point(coord[1], coord[2])
                    if boundary.contains(point):
                        good_inds.append(ind)
                rx_coords = rx_coords[np.array(good_inds)]
            else:
                assert(len(boundary) == 2)
                lat_min, lon_min = boundary.min(axis=0)
                lat_max, lon_max = boundary.max(axis=0)
                rx_coords = rx_coords[ ~np.isinf(rx_coords[:,0])]
                rx_coords = rx_coords[ ~(rx_coords[:,1] == 0)]
                rx_coords = rx_coords[ ~(rx_coords[:,1] < lat_min)]
                rx_coords = rx_coords[ ~(rx_coords[:,1] > lat_max)]
                rx_coords = rx_coords[ ~(rx_coords[:,2] < lon_min)]
                rx_coords = rx_coords[ ~(rx_coords[:,2] > lon_max)]
            if tx_coords is None:
                return rx_coords
        return tx_coords, rx_coords
    

    def get_citation(self):
        label = f'Loading data from DS {self.params.dataset_index}\nPlease reference the original work when using this dataset:\n'
        if self.params.dataset_index == 1:
            citation = f"""@article{{patwari2003relative,
  title={{Relative location estimation in wireless sensor networks}},
  author={{Patwari, Neal and Hero, Alfred O and Perkins, Matt and Correal, Neiyer S and {{O'Dea}}, Robert J}},
  journal={{IEEE Transactions on Signal Processing}},
  volume={{51}},
  number={{8}},
  pages={{2137--2148}},
  year={{2003}},
  publisher={{IEEE}}
}}"""
            license = 'Data used by permission under CC BY 4.0 License.'
            url = 'https://dx.doi.org/10.15783/C7630J'
        elif self.params.dataset_index in [2,3,4]:
            citation = f"""@inproceedings{{sarkar2020llocus,
  title={{LLOCUS: learning-based localization using crowdsourcing}},
  author={{Sarkar, Shamik and Baset, Aniqua and Singh, Harsimran and Smith, Phillip and Patwari, Neal and Kasera, Sneha and Derr, Kurt and Ramirez, Samuel}},
  booktitle={{Proceedings of the 21st International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing}},
  pages={{201--210}},
  year={{2020}}
}}"""
        elif self.params.dataset_index == 5:
            citation = f"""@inproceedings{{mitchell2022tldl,
  title={{Deep Learning-based Localization in Limited Data Regimes}},
  author={{Mitchell, Frost and Baset, Aniqua and Patwari, Neal and Kasera, Sneha Kumar and Bhaskara, Aditya}},
  booktitle={{Proceedings of the 2022 ACM Workshop on Wireless Security and Machine Learning}},
  pages={{15--20}},
  year={{2022}}
}}"""
        elif self.params.dataset_index == 6:
            citation = f"""@INPROCEEDINGS{{mitchell2023cutl,
  author={{Mitchell, Frost and Patwari, Neal and Bhaskara, Aditya and Kasera, Sneha Kumar}},
  booktitle={{2023 20th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)}}, 
  title={{Learning-based Techniques for Transmitter Localization: A Case Study on Model Robustness}}, 
  year={{2023}},
  volume={{}},
  number={{}},
  pages={{133-141}},
  keywords={{Location awareness;Training;Wireless sensor networks;Wireless networks;Radio transmitters;Receivers;Interference;transmitter localization;model robustness;RF spectrum sensing}},
  doi={{10.1109/SECON58729.2023.10287483}}}}
"""
            license = 'Data used by permission under CC BY 4.0 License.'
            url = 'https://doi.org/10.5281/zenodo.7259895'
        elif self.params.dataset_index == 7:
            citation =f"""@dataset{{aernouts_2018_1193563,
  author       = {{Aernouts, Michiel and
                  Berkvens, Rafael and
                  Van Vlaenderen, Koen and
                  Weyn, Maarten}},
  title        = {{{{Sigfox and LoRaWAN Datasets for Fingerprint 
                   Localization in Large Urban and Rural Areas}}}},
  month        = mar,
  year         = 2018,
  publisher    = {{Zenodo}},
  version      = {{1.0}},
  doi          = {{10.5281/zenodo.1193563}},
  url          = {{https://doi.org/10.5281/zenodo.1193563}}
}}
"""
            license = 'Data used by permission under CC BY 4.0 License.'
            url = 'https://doi.org/10.5281/zenodo.1193563'
        elif self.params.dataset_index == 8:
            citation = f"""@misc{{tadikmeas2024,
    author       = "Tadik, S. and Singh, A. and Mitchell, F. and Hu, Y. and Yao, X. and Webb, K. and Sarbhai, A. and Maas, D. and Orange, A. and Van der Merwe, J. and Patwari, N. and Ji, M. and Kasera, Sneha K. and Bhaskara, A. and Durgin, Gregory D.",
    title        = "Salt Lake City 3534 MHz Multi-Transmitter Measurement Campaign",
    year         = "2024",
    month        = "March",
    howpublished = {{\\url{{https://github.com/serhatadik/slc-3534MHz-meas}}}}
}}"""
            license = 'Data used by permission under MIT License'
            url = 'https://github.com/serhatadik/slc-3534MHz-meas'
        
        full_reference = f"{label}\n{citation}\n{license}\n{url}\n"
        return full_reference


    def load_data(self):
        data_list = []
        #print(self.get_citation()) # TODO need to fix citation for dataset 9
        if self.params.dataset_index == 1:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                rx_loc_inds = {}
                current_ind = 0
                for line in lines:
                    columns = line.split()
                    num_trans = int(columns[1])
                    tx_locs = []
                    tx_gains = []
                    for i in range(num_trans):
                        x, y = columns[2+i].split(',')
                        tx_locs.append([ round(float(x)*2) / 2, round(float(y)*2)/2])
                        tx_gains.append(1)
                    rx_tups = []
                    for rx in columns[2+num_trans:]:
                        rss, x, y = rx.split(',')
                        if (x,y) in rx_loc_inds:
                            ind = rx_loc_inds[(x,y)]
                        else:
                            rx_loc_inds[(x,y)] = current_ind
                            ind = current_ind
                            current_ind += 1
                        rx_tups.append([float(rss), float(x), float(y), ind])
                    print(np.array(rx_tups)[:,1:].min(axis=0))
                    min_coords = np.array([0,5,1,0])
                    data_list.append([np.array(tx_locs) + min_coords[1:3], np.array(rx_tups) + min_coords, np.array(tx_gains)])
                self.location_index_dict = rx_loc_inds
        elif self.params.dataset_index < 5:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    columns = line.split()
                    num_trans = int(columns[1])
                    tx_locs = []
                    tx_gains = []
                    for i in range(num_trans):
                        x, y = columns[2+i].split(',')
                        tx_locs.append([ round(float(x)*2) / 2, round(float(y)*2)/2])
                        tx_gains.append(1)
                    rx_tups = []
                    for rx in columns[2+num_trans:]:
                        rss, x, y = rx.split(',')
                        rx_tups.append([float(rss), float(x), float(y)])
                    data_list.append([np.array(tx_locs), np.array(rx_tups), np.array(tx_gains)])
                max_rx = max([len(ent[1]) for ent in data_list])
                self.location_index_dict = dict(zip(range(max_rx), range(max_rx)))
        elif self.params.dataset_index == 5:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                preamble = ''
                for line in lines:
                    first_length = len(line.split(',')[0])
                    this_preamble = line[first_length:].split('-')[0]
                    if this_preamble == preamble:
                        continue
                    preamble = this_preamble
                    columns = line.split(',')
                    num_trans = int(columns[1])
                    column_index = 2
                    tx_locs = []
                    tx_gains = []
                    for i in range(num_trans):
                        rss, x, y = columns[column_index:column_index + 3]
                        tx_locs.append([float(x)-1, float(y)-1])
                        tx_gains.append(rss)
                        column_index += 3
                    if self.params.one_tx and (len(tx_locs) > 1):
                        continue
                    rx_tups = []
                    while column_index < len(columns):
                        rss, x, y = columns[column_index:column_index +3]
                        rx_tups.append([float(rss), float(x)-1, float(y)-1])
                        column_index += 3
                    data_list.append([np.array(tx_locs), np.array(rx_tups), np.array(tx_gains)])
                max_rx = max([len(ent[1]) for ent in data_list])
                self.location_index_dict = dict(zip(range(max_rx), range(max_rx)))
        elif self.params.dataset_index == 6: #Loading our powder data
            tx_metadata = []
            location_25p_dict = {}
            location_100p_dict = {}
            tx_dicts = {}
            with open('datasets/frs_data/location_indexes.json', 'r') as f:
                location_index_dict = json.load(f)
            bus_indexes = [location_index_dict[k] for k in location_index_dict if 'bus-' in k]
            bus_rss = {ind:[] for ind in bus_indexes}
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    tmp_tx = json.load(f)
                tx_dicts = {**tx_dicts, **tmp_tx}
            for key in tx_dicts:
                arr = np.array([line[:3] + [location_index_dict[line[3]]] for line in tx_dicts[key]['rx_data']])
                for ind in bus_indexes:
                    if ind in arr[:,3]:
                        bus_rss[ind].append(arr[arr[:,3] == ind,0][0])
                tx_dicts[key]['rx_data'] = arr
                if 'tx_coords' in tx_dicts[key]:
                    tx_dicts[key]['tx_coords'] = np.array(tx_dicts[key]['tx_coords'])
                    for one_metadata in tx_dicts[key]['metadata']:
                        one_metadata['time'] = key
                        if 'precip' not in one_metadata:
                            one_metadata['precip'] = 'none'
            default_min_rss = {2: -96, 3: -100, 4: -95}
            default_max_rss = {2: -35, 3: 20, 4: -4}
            for name in location_index_dict:
                key = location_index_dict[name]
                rx_type = 1 if 'bus-' in name else 2 if 'cnode' in name else 3 if ('cbrs' in name or 'cell' in name) else 4
                if rx_type == 1 and len(bus_rss[key]) > 0:
                    location_25p_dict[key] = np.quantile(bus_rss[key], 0.25)
                    location_100p_dict[key] = max(bus_rss[key])
                elif rx_type != 1:
                    location_25p_dict[key] = default_min_rss[rx_type]
                    location_100p_dict[key] = default_max_rss[rx_type]
            self.location_min_rss_dict = location_25p_dict
            self.location_max_rss_dict = location_100p_dict
            self.load_geotiff('datasets/frs_data/corrected_dsm.tif', 'datasets/frs_data/corrected_buildings.tif') # Img is loaded as a rasterio image, with coordinates in UTM zone 12
            boundary_gps_coordinates = coordinates.CAMPUS_LATLON
            if len(boundary_gps_coordinates) == 4:
                bounds = Polygon(boundary_gps_coordinates)
            else:
                bounds = boundary_gps_coordinates
            stationary_dict = {}
            off_campus_dict = {}
            for key in tx_dicts:
                if 'tx_coords' not in tx_dicts[key]: continue
                tx_locs = tx_dicts[key]['tx_coords']
                if len(tx_locs) != 1 and self.params.one_tx: continue
                rx_tups = tx_dicts[key]['rx_data']
                tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs, rx_coords=rx_tups)
                if len(tx_locs) == 0:
                    off_campus_dict[key] = tx_dicts[key]
                    continue
                # Here, we're adjusting the utm coordinates to a local system, just because we have some precision problems at 32 bits where the affine transform takes place.
                tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin)
                tx_gains = [1] * len(tx_locs)
                rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                if 'stationary' in tx_dicts[key]['metadata'][0]:
                    stationary_dict.setdefault(tx_dicts[key]['metadata'][0]['stationary'], []).append([tx_locs, rx_tups[:,:3], tx_gains, tx_dicts[key]['metadata'], rx_tups[:,3]])
                    continue
                data_list.append([tx_locs, rx_tups, tx_gains])
                #tx_dicts[key]['metadata'][0]['time'] = key
                tx_metadata.append(tx_dicts[key]['metadata'])
                #location_inds.append(rx_tups[:,3])
            for stationary_index in stationary_dict:
                continue
                tx_locs = np.array( [entry[0] for entry in stationary_dict[stationary_index] ] ).mean(axis=0)
                rx_tups = np.concatenate( [entry[1] for entry in stationary_dict[stationary_index] ] )
                tx_gains = [1]
                tx_md = stationary_dict[stationary_index][0][3]
                loc_inds = stationary_dict[stationary_index][0][4]
                data_list.append([tx_locs, rx_tups, tx_gains])
                tx_metadata.append(tx_md)
                location_inds.append(loc_inds)
            if False: #not self.params.force_num_tx: # This should execute when we want empty with no tx. Should be fixed.
                for key in no_tx_dicts:
                    tx_locs = []
                    tx_gains = []
                    rx_tups = no_tx_dicts[key]['rx_data']
                    rx_tups = self.filter_bounds(bounds, rx_coords=rx_tups)
                    rx_tups = rx_tups[ ~np.isinf(rx_tups[:,0])]
                    rx_tups = rx_tups[ ~(rx_tups[:,1] == 0)]
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    data_list.append([tx_locs, rx_tups, tx_gains])
                    tx_metadata.append([{}])
                    #location_inds.append(rx_tups[:,3])
            self.location_index_dict = {}
            for key in location_index_dict:
                self.location_index_dict[location_index_dict[key]] = key
            #self.location_inds = np.array(location_inds)
        elif self.params.dataset_index == 7: #Loading antwerp lorawan data
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                with open('datasets/data_antwerp/chosen_gateways_5_sensors.txt') as f:
                    locs = json.load(f)
                self.load_geotiff('datasets/data_antwerp/antwerp_zoom_dsm.tif') # Img is loaded as a rasterio image, with coordinates in UTM zone 12
                tx_metadata = []
                used_ids = {}
                id_index = 0
                boundary_gps_coordinates = coordinates.ANTWERP_LATLON
                if len(boundary_gps_coordinates) == 4:
                    bounds = Polygon(boundary_gps_coordinates)
                else:
                    bounds = boundary_gps_coordinates
                for sample in data:
                    stationary_dict = {}
                    if sample['hdop'] > 2:
                        continue
                    metadata = {'hdop':sample['hdop'], 'sf': sample['sf']}
                    tx_locs = np.array([[sample['latitude'], sample['longitude'] ]])

                    if len(tx_locs) != 1 and self.params.one_tx: continue
                    rx_tups = []
                    gateway_ids = []
                    for gateway in sample['gateways']:
                        id = gateway['id']
                        if id not in locs: continue
                        #new_lat = coordinates.convert_gps_to_utm( np.array( [[locs[id]['latitude'], locs[id]['longitude']] ] ), origin_to_subtract=origin)
                        rx_tups.append( [ gateway['rssi'], locs[id]['latitude'], locs[id]['longitude'] ] )
                        gateway_ids.append(id)
                        metadata['time'] = gateway['rx_time']['time'][:10]
                    if len(rx_tups) == 0 or len(rx_tups) < self.params.min_sensors:
                        continue
                    for id in gateway_ids:
                        if id not in used_ids:
                            used_ids[id] = id_index
                            id_index += 1
                    for i in range(len(rx_tups)):
                        rx_tups[i].append(used_ids[gateway_ids[i] ])
                    rx_tups = np.array(rx_tups)
                    print(f"GPS tx_locs {tx_locs} and rx_tups {rx_tups}")

                    tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs, rx_coords=rx_tups)
                    if len(tx_locs) == 0 or len(rx_tups) == 0 or len(rx_tups) < self.params.min_sensors:
                        continue
                    tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin)
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    tx_gains = [1] * len(tx_locs)
                    print(f"UTM tx_locs {tx_locs} and rx_tups {rx_tups}")
                    print(f"tx_gains {tx_gains}")

                    data_list.append( [tx_locs, rx_tups, tx_gains] )
                    tx_metadata.append(metadata)
                self.location_index_dict = {}
                for key in used_ids:
                    self.location_index_dict[ used_ids[key]] = key
        elif self.params.dataset_index == 8:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                tx_metadata = []
                used_ids = {}
                id_index = 0
                self.load_geotiff('datasets/frs_data/corrected_dsm.tif', 'datasets/frs_data/corrected_buildings.tif') # Img is loaded as a rasterio image, with coordinates in UTM zone 12
                boundary_gps_coordinates = coordinates.DENSE_LATLON
                if len(boundary_gps_coordinates) == 4:
                    bounds = Polygon(boundary_gps_coordinates)
                else:
                    bounds = boundary_gps_coordinates
                new_id = 0
                for sample in data:
                    metadata = {'time':sample, 'mobility': data[sample]["metadata"]}
                    lats = [arr[3] for arr in data[sample]["pow_rx_tx"]]
                    for lat in lats:
                        if lat not in used_ids:
                            used_ids[lat] = new_id
                            new_id += 1
                    rx_tups = np.array([[arr[0], arr[3], arr[4], used_ids[arr[3]]] for arr in data[sample]["pow_rx_tx"]])
                    tx_locs = np.array(data[sample]["pow_rx_tx"][0][1:3])
                    tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs[None], rx_coords=rx_tups)
                    if len(tx_locs) == 0: continue
                    tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin)
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    tx_gains = [1] * len(tx_locs)
                    data_list.append( [tx_locs, rx_tups, tx_gains] )
                    tx_metadata.append(metadata)
                self.location_index_dict = {}
                location_names = [
                    'cnode-ebc-dd-b210',
                    'cnode-guesthouse-dd-b210',
                    'cnode-mario-dd-b210',
                    'cnode-moran-dd-b210',
                    'cnode-wasatch-dd-b210',
                    'cbrssdr1-ustar-comp'
                ]
                for key, name in zip(used_ids, location_names):
                    self.location_index_dict[ used_ids[key]] = name
        elif self.params.dataset_index == 9: # helium_SD dataset
            for data_file in self.data_files:
                # with open(data_file, 'r') as f:
                #     print(f)
                #     data = json.load(f)
                if self.params.data_filename:
                    filepath = self.params.data_filename
                else:
                    filepath = ds9_path
                with open(filepath) as f:
                    #----------------------- from dataset < 5
                    lines = f.readlines()
                    #rx_loc_inds = {}
                    #current_ind = 0
                new_id = 0
                used_ids = {}

                # set data filter boundaries
                if self.params.coordinates:
                    boundary_gps_coordinates = self.params.coordinates
                else:
                    boundary_gps_coordinates = coordinates.HELIUMSD_LATLON # TODO: may need to fix this (Bottom left, top right?)
                if len(boundary_gps_coordinates) == 4:
                    bounds = Polygon(boundary_gps_coordinates)
                else:
                    bounds = boundary_gps_coordinates

                for line in lines[1:]: # one tx-rx pair per line, skip first line because headers
                    columns = line.strip().split(',')
                    #num_trans = int(columns[1])
                    #print(columns)
                    tx_locs = np.array([[float(columns[1]), float(columns[2])]])
                    #if len(tx_locs) != 1 and self.params.one_tx: continue # not really needed here because its CSV
                    #tx_locs = []
                    #tx_gains = []
                    # for i in range(num_trans):
                    #     x, y = columns[2+i].split(',')
                    #     tx_locs.append([ round(float(x)*2) / 2, round(float(y)*2)/2])
                    #     tx_gains.append(1)


                    # see if we want to skip this RXer
                    rx_lat = float(columns[3])

                    # rx_blacklist = ['47.50893434', '47.51250611', '47.56667388', '47.61922143', '47.4692100571904',
                    #  '47.47742889862316', '47.48925703694905', '47.50359742681539', '47.50840335356933',
                    #  '47.508746888592974', '47.51072216107213', '47.51731330472538', '47.52369040459335',
                    #  '47.530655757870726', '47.53205816773291', '47.53273388440102', '47.53393616926552',
                    #  '47.53500568388223', '47.538850220954174', '47.54240398585348', '47.542722886879794',
                    #  '47.54327270368826', '47.54468061147836', '47.54823440420244', '47.54837431128243',
                    #  '47.553193946170445', '47.55351181613522', '47.554626222317054', '47.55472254606462',
                    #  '47.55602653173352', '47.55647317987443', '47.559019618951936', '47.56965033955307',
                    #  '47.56985542720164', '47.57011064833567', '47.57016078334357', '47.570418274093385',
                    #  '47.57229596451633', '47.61067184190455', '47.61440590930977', '47.61536982933476',
                    #  '47.61582542823447', '47.61707073349552', '47.61717326802679', '47.61749062748967',
                    #  '47.61771134123786', '47.61820102339988', '47.624933594863656', '47.62788464575477',
                    #  '47.6279178754742', '47.62798810555178', '47.62891578888903', '47.628999870304085',
                    #  '47.63317115265743', '47.63321182606039', '47.63370215464106', '47.63370706366582',
                    #  '47.63733751209776', '47.64030456039423', '47.64248991491972', '47.64694171272788',
                    #  '47.65073198859608', '47.65390855096642', '47.57920936741801', '47.58019670193037',
                    #  '47.58049332880381', '47.58292549332933', '47.585471925347406', '47.58853890118896',
                    #  '47.58902400026363', '47.59340963652312', '47.594241262487685', '47.59864847808752',
                    #  '47.60104020422019', '47.60566438533686', '47.60595473978473', '47.60637337907797',
                    #  '47.60637979517027', '47.609964745154365'] # 25th percentile RXers

                    rx_blacklist = ['47.50893434', '47.51250611', '47.61922143', '47.4692100571904', '47.47742889862316',
                     '47.47922825256919', '47.486512396667486', '47.50359742681539', '47.50840335356933',
                     '47.51072216107213', '47.51731330472538', '47.530655757870726', '47.53205816773291',
                     '47.53273388440102', '47.53393616926552', '47.538850220954174', '47.542722886879794',
                     '47.54327270368826', '47.54468061147836', '47.54580855981581', '47.55104682159288',
                     '47.553193946170445', '47.55472254606462', '47.56965033955307', '47.56985542720164',
                     '47.57011064833567', '47.57016078334357', '47.570418274093385', '47.57229596451633',
                     '47.61385761645424', '47.61531726651759', '47.61703782509865', '47.61707073349552',
                     '47.61771134123786', '47.61820102339988', '47.624933594863656', '47.62788464575477',
                     '47.6279178754742', '47.62792341916671', '47.62798810555178', '47.628999870304085',
                     '47.63317115265743', '47.63321182606039', '47.63370215464106', '47.63370706366582',
                     '47.63733751209776', '47.64694171272788', '47.65073198859608', '47.65390855096642',
                     '47.58049332880381', '47.58292549332933', '47.585471925347406', '47.58853890118896',
                     '47.58902400026363', '47.58944644884187', '47.59340963652312', '47.59864847808752',
                     '47.60566438533686', '47.60595473978473', '47.60637337907797', '47.609964745154365'] #10th

                    if self.params.rx_blacklist:
                        if str(rx_lat) in self.params.rx_blacklist:
                            #print(f"Skipping RX node {rx_lat}")
                            continue

                    # if str(rx_lat) in rx_blacklist: # skip this RX
                    #     #print(f"Skipping RX node {rx_lat}")
                    #     continue

                    # to generate TX IDs; tx lat rounded to 3 dec places to decide unique TXers, diff of .001 ~= 100m
                    round_lat = float(round(float(columns[1]),3))
                    if round_lat not in used_ids:
                        used_ids[round_lat] = new_id
                        new_id += 1

                    # rssi, rx_lat, rx_lon, used_id
                    rx_tups = np.array([[float(columns[6]), rx_lat, float(columns[4]), used_ids[round_lat]]])
                    #print(f"GPS tx_locs {tx_locs} and rx_tups {rx_tups}")

                    # for rx in columns[2+num_trans:]:
                    #     rss, x, y = rx.split(',')
                    #     rx_tups.append([float(rss), float(x), float(y)])




                    # filter and convert
                    #tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs, rx_coords=rx_tups) # TODO: fix bounds
                    #tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin) # TODO
                    tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=None)
                    #rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=None)
                    #print(f"UTM tx_locs {tx_locs} and rx_tups {rx_tups}")

                    tx_gains = [1] * len(tx_locs)
                    #print(f"tx_gains {tx_gains}")

                    data_list.append( [tx_locs, rx_tups, tx_gains] )

                # consolidate RX measurements per receiver instead of line by line
                data_list = consolidate_data(data_list, precision = 2)

                self.location_index_dict = {}
                for key in used_ids:
                    self.location_index_dict[ used_ids[key]] = key
            # end of dataset_index == 9 ## helium_SD dataset


        tx_vecs = []
        rx_vecs = []
        #tx_gains = []

        min_x = 1e9
        min_y = 1e9
        max_x = -1e9
        max_y = -1e9
        min_rss = 1e9
        max_rss = -1e9
        #all_tx = []
        #all_rx = []
        #tx_count = []
        #rx_count = []
        for entry in data_list:
        #    if len(entry[0]) == 3 and [-1.0, -1.0] in entry[0]:
        #        continue
        #    tx_count.append(len(entry[0]))
            for tx in entry[0]:
        #        all_tx.append(tx)
                min_x = min(min_x, tx[0])
                min_y = min(min_y, tx[1])
                max_x = max(max_x, tx[0])
                max_y = max(max_y, tx[1])
        #    rx_count.append(len(entry[1]))
            for rx in entry[1]:
        #        all_rx.append(rx)
                min_x = min(min_x, rx[1])
                min_y = min(min_y, rx[2])
                max_x = max(max_x, rx[1])
                max_y = max(max_y, rx[2])
                min_rss = min(min_rss, rx[0])
                max_rss = max(max_rss, rx[0])
        #if self.verbose:
        #    print('X Range: [%i:%i]\tY Range: [%i:%i]\tRSS Range:[%.1f:%.1f]' % (min_x, min_y, max_x, max_y, min_rss, max_rss))
        self.min_x   =  min_x
        self.min_y   = min_y 
        self.max_x   = max_x 
        self.max_y   = max_y 
        self.noise_floor = -114
        self.min_rss = min(self.noise_floor, min_rss)
        transmit_power = 5
        self.max_rss = max(max_rss, transmit_power)
        ## Meter scale of 0.5 corresponds to 0.5 meters per pixel
        #if self.forced_meter_scale:
        #    self.meter_scale = self.forced_meter_scale
        #    self.img_size_x = int((max_x - min_x) / self.meter_scale) + 2*self.buffer
        #    self.img_size_y = int((max_y - min_y) / self.meter_scale) + 2*self.buffer
        #    self.img_size = max(self.img_size_x, self.img_size_y)
        #elif self.forced_img_size:
        #    self.img_size = self.forced_img_size
        #    x_scale = (max_x - min_x) / (self.img_size - 2*self.buffer)
        #    y_scale = (max_y - min_y) / (self.img_size - 2*self.buffer)
        #    self.meter_scale = max(x_scale, y_scale)
        #else:
        #    self.meter_scale = 1
        #    self.img_size_x = int((max_x - min_x) / self.meter_scale) + 2*self.buffer
        #    self.img_size_y = int((max_y - min_y) / self.meter_scale) + 2*self.buffer
        #    self.img_size = max(self.img_size_x, self.img_size_y)
        #    while self.img_size < 30:
        #        self.meter_scale = .5 * self.meter_scale
        #        self.img_size = int(max((max_x - min_x) / self.meter_scale, (max_y - min_y) / self.meter_scale ) + 2*self.buffer)
        #self.img_size_x = self.img_size
        #self.img_size_y = self.img_size
        #min_tx_arr = np.array([min_x, min_y])
        min_rx_arr = np.array([self.noise_floor, 0, 0, 0])
        for entry in data_list:
            if len(entry[0]) == 3 and [-1.0, -1.0] in entry[0]:
                continue
            if len(entry[0]) == 0:
                continue
                tx_vecs.append( np.array(entry[0])) 
            else:
                #tx_vecs.append( self.buffer + (np.array(entry[0]) - min_tx_arr)/ self.meter_scale) 
                tx_vecs.append( np.array(entry[0])) 
            #rx_vecs.append( np.array([0, self.buffer, self.buffer]) + (np.array(entry[1]) - min_rx_arr)/ np.array([self.max_rss - self.noise_floor, self.meter_scale, self.meter_scale])) 
            if self.params.dataset_index < 6 or self.params.dataset_index == 8:
                entry_min_rss = min_rss
                entry_max_rss = max_rss
            elif self.params.dataset_index == 6:
                entry_min_rss = np.array([self.location_min_rss_dict[int(loc_ind)] for loc_ind in entry[1][:,3] ])
                entry_max_rss = np.array([self.location_max_rss_dict[int(loc_ind)] for loc_ind in entry[1][:,3] ])
            elif self.params.dataset_index == 7:
                entry_min_rss = -126
                entry_max_rss = -70
            elif self.params.dataset_index == 9:
                entry_min_rss = -136
                entry_max_rss = -45
            rx_vec = np.zeros((len(entry[1]),5))
            #rx_vec[:,0] = entry[1][:,0]/ np.array([self.max_rss - self.noise_floor, 1, 1, 1])
            rx_vec[:,0] = (entry[1][:,0] - entry_min_rss)/ (entry_max_rss - entry_min_rss)
            if self.params.dataset_index == 1:
                rx_vec[:,1:4] = entry[1][:,1:]
            elif self.params.dataset_index < 6:
                rx_vec[:,1:3] = entry[1][:,1:]
            elif self.params.dataset_index == 8:
                rx_vec[:,1:4] = entry[1][:,1:]
                rx_vec[:,4] = 2
            elif self.params.dataset_index == 9:
                rx_vec[:,1:4] = entry[1][:,1:]
                rx_vec[:,4] = 2
            elif self.params.dataset_index >= 6:
                rx_vec[:,1:4] = entry[1][:,1:]
                location_names = [self.location_index_dict[id] for id in rx_vec[:,3]]
                rx_vec[:,4] = np.array([1 if 'bus-' in name else 2 if 'cnode' in name else 3 if 'cbrs' in name else 4 for name in location_names])
                if self.params.remove_mobile:
                    rx_vec = rx_vec[rx_vec[:,4] != 1]
                # Make common inds
            ### Filter buses because they might be causing more trouble than they are helping with...
            #rx_vec = rx_vec[rx_vec[:,4] != 1]

            rx_vecs.append(rx_vec) 
            #tx_gains.append(entry[2])
            #self.y_lengths.append(len(entry[0]) )
            #self.x_lengths.append(len(entry[1]) )
        #Setting
        self.max_num_rx = len(self.location_index_dict)
        if self.params.dataset_index == 6:
            self.all_min_rss = np.array([self.location_min_rss_dict[key] for key in sorted(self.location_min_rss_dict)])
            self.all_max_rss = np.array([self.location_max_rss_dict[key] if key in self.location_max_rss_dict else -20 for key in sorted(self.location_min_rss_dict)])
        else:
            self.all_min_rss = self.min_rss
            self.all_max_rss = self.max_rss
        if self.params.dataset_index in [6,7,8]:
            self.data[None] = self.Samples(self, np.array(rx_vecs, dtype=object), np.array(tx_vecs, dtype=object), tx_metadata=np.array(tx_metadata, dtype=object))
        else:
            self.data[None] = self.Samples(self, np.array(rx_vecs, dtype=object), np.array(tx_vecs, dtype=object))
        samples = self.data[None]
        origin = samples.origin
        top_corner = origin + np.array([samples.rectangle_width,  samples.rectangle_height])
        self.corners = np.array([origin, top_corner])


    def get_min_max_rss_from_key(self, key):
        if hasattr(self, 'location_min_rss_dict'):
            min_rss = self.location_min_rss_dict[key]
        else:
            min_rss = self.min_rss
        if hasattr(self, 'location_max_rss_dict'):
            max_rss = self.location_max_rss_dict[key] if key in self.location_max_rss_dict else -20
        else:
            max_rss = self.max_rss
        return min_rss, max_rss


    def load_geotiff(self, img_file, building_file=None):
        self.elevation_map = rasterio.open(img_file)
        self.elevation_map.origin = np.array(self.elevation_map.transform)[:6].reshape(2,3) @ np.array([0, self.elevation_map.shape[0], 1])
        if building_file is not None and os.path.exists(building_file):
            self.building_map = rasterio.open(building_file)
        else:
            self.building_map = None

        
    def separate_random_data(self, data_source_key=None, test_size=0.2, train_size=None, data_key_prefix='', use_folds=False, n_splits=5, num_tx=None, ending_keys=['train', 'test'], random_state=None):
        if isinstance(num_tx, int):
            num_tx = [num_tx]
        if self.params.one_tx:
            num_tx = [1]
        data_source = self.data[data_source_key]
        if self.params.dataset_index == 6:
            inds = self.filter_inds_by_metadata(source_key=data_source_key)
            x_vecs = copy.deepcopy(data_source.rx_vecs[inds])
            y_vecs = copy.deepcopy(data_source.tx_vecs[inds])
            tx_metadata = copy.deepcopy(data_source.tx_metadata[inds])
        else:
            x_vecs = copy.deepcopy(data_source.rx_vecs)
            y_vecs = copy.deepcopy(data_source.tx_vecs)

        if num_tx == None or len(num_tx) < 1:
            num_tx = list(range(data_source.max_num_tx+1))
        valid_inds = [i for i, vec in enumerate(y_vecs) if len(vec) in num_tx]
        x_vecs = x_vecs[valid_inds]
        y_vecs = y_vecs[valid_inds]

        count = 0
        unique_locations_dict = {}
        empty_count = 0
        for i, y_vec in enumerate(y_vecs):
            if len(y_vec) == 0: 
                key = empty_count 
                empty_count += 1
            else:
                key = y_vec[:,:2].tobytes()
            if key in unique_locations_dict:
                unique_locations_dict[key].append(i)
                count += 1
            else:
                unique_locations_dict[key] = [i]
                count += 1

        if use_folds:
            print('Not implemented.')
            #kf = KFold(n_splits=n_splits, random_state=self.random_state, shuffle=True)
            #fold_counter = 0
            #for train_index, test_index in kf.split(x_lengths):
            #    x_vecs_train, x_vecs_test = x_vecs[train_index], x_vecs[test_index]
            #    y_vecs_train, y_vecs_test = y_vecs[train_index], y_vecs[test_index]
            #    x_lengths_train, x_lengths_test = x_lengths[train_index], x_lengths[test_index]
            #    y_lengths_train, y_lengths_test = y_lengths[train_index], y_lengths[test_index]
            #    self.make_data_entry(data_key_prefix+'_other_folds_%i' % fold_counter, x_train, y_train, x_vecs_train, y_vecs_train, x_lengths_train, y_lengths_train, make_tensors=make_tensors)
            #    self.make_data_entry(data_key_prefix+'_fold_%i' % fold_counter, x_test, y_test, x_vecs_test, y_vecs_test, x_lengths_test, y_lengths_test, make_tensors=make_tensors)
            #    fold_counter += 1
        else:
            if len(data_key_prefix) == 0:
                train_key = ending_keys[0]
                test_key = ending_keys[1]
            else:
                train_key = data_key_prefix+'_' + ending_keys[0]
                test_key = data_key_prefix+'_' + ending_keys[1]
            
            train_location_keys, test_location_keys = train_test_split(list(unique_locations_dict.keys()), shuffle=True, test_size=test_size, train_size=train_size, random_state=random_state if random_state is not None else self.params.random_state)
            train_inds = []
            test_inds = []
            for key in train_location_keys:
                train_inds += unique_locations_dict[key]
            for key in test_location_keys:
                test_inds += unique_locations_dict[key]

            x_vecs_train, x_vecs_test = x_vecs[train_inds], x_vecs[test_inds]
            label_train, label_test = y_vecs[train_inds], y_vecs[test_inds]
            if self.params.dataset_index == 6:
                metadata_train, metadata_test = tx_metadata[train_inds], tx_metadata[test_inds]
                self.data[train_key] = self.Samples(self, x_vecs_train, label_train, tx_metadata=metadata_train)
                self.data[test_key] = self.Samples(self, x_vecs_test, label_test, tx_metadata=metadata_test)
            elif self.params.dataset_index == 7:
                self.data[train_key] = self.Samples(self, x_vecs_train, label_train, no_shift=True)
                self.data[test_key] = self.Samples(self, x_vecs_test, label_test, no_shift=True)
            else:
                self.data[train_key] = self.Samples(self, x_vecs_train, label_train)
                self.data[test_key] = self.Samples(self, x_vecs_test, label_test)
            return train_key, test_key


    def set_default_keys(self, aug_train_key=None, train_key=None, test_key=None, test_keys=[]):
        self.aug_train_key = aug_train_key if aug_train_key!=None or not hasattr(self, 'aug_train_key') else self.aug_train_key
        self.train_key = train_key if train_key!=None or not hasattr(self, 'train_key') else self.train_key
        self.test_key = test_key if test_key!=None or not hasattr(self, 'test_key') else self.test_key
        self.test_keys = test_keys if len(test_keys)!=0 or not hasattr(self, 'test_keys') else self.test_keys
        keys =  set([aug_train_key, train_key, test_key] + test_keys)
        for key in keys:
            if key is not None and not hasattr(self.data[key],'dataloader'):
                self.data[key].make_tensors()


    def img_height(self): 
        if hasattr(self, 'train_key'):
            key = self.train_key
        else:
            key = None
        return round(self.data[key].rectangle_height / self.params.meter_scale) + 1


    def img_width(self):
        if hasattr(self, 'train_key'):
            key = self.train_key
        else:
            key = None
        return round(self.data[key].rectangle_width / self.params.meter_scale) + 1


    def make_filtered_sample_source(self, filter_boundaries, data_key, source_key=None, convert_to_utm=True):
        source = self.data[source_key]
        #if np.where((-180 < coords) and (coords < 180), True, False).prod(): #This is a check to see if number is latlon or UTM coordinate. Not very safe.
        if convert_to_utm:
            for i, filter_boundary in enumerate(filter_boundaries):
                filter_boundaries[i] = coordinates.convert_gps_to_utm(filter_boundary, origin_to_subtract=self.elevation_map.origin)
                
        self.data[data_key] = self.Samples(self, rx_vecs=source.rx_vecs, tx_vecs=source.tx_vecs, filter_boundaries=filter_boundaries, tx_metadata=source.tx_metadata if hasattr(source, 'tx_metadata') else None)

    
    def make_missing_samples(self, filtered_key, source_key=None):
        tx_vecs = []
        rx_vecs = []
        tx_metadata = []
        source_samples = self.data[source_key]
        filtered_samples = self.data[filtered_key]
        for i, tx in enumerate(source_samples.tx_vecs):
            if tx in filtered_samples.tx_vecs: continue
            tx_vecs.append(tx)
            rx_vecs.append(source_samples.rx_vecs[i])
            tx_metadata.append(source_samples.tx_metadata[i])
        samp = self.Samples(self, rx_vecs=rx_vecs, tx_vecs=tx_vecs, tx_metadata=tx_metadata)
        return samp


    def print_dataset_stats(self):
        print('Pixel Scale (m):', self.params.meter_scale)
        print('Image height:', self.img_height(), self.img_width())
        print('Meter Scale:', self.params.meter_scale)
        for key in self.data.keys():
            if key is not None and 'synthetic' in key:
                y_lengths = self.data[key].ordered_dataloader.dataset.tensors[1][:,:,0].sum(axis=1).cpu().numpy()
                x_lengths = (self.data[key].ordered_dataloader.dataset.tensors[0][:,:,3] != 0).sum(axis=1).cpu().numpy()
            else:
                y_lengths = [len(vec) for vec in self.data[key].tx_vecs]
                x_lengths = [len(vec) for vec in self.data[key].rx_vecs]
            print('Dataset %s' % key)
            print('  Length:' , len(y_lengths))
            print('  Number of transmitters:', np.unique(y_lengths, return_counts=True) )
            print('  Number of sensors in dataset:', np.unique(x_lengths, return_counts=True) )


    def get_rx_data_by_tx_location(self, source_key=None, combine_sensors=True, use_db_for_combination=True, required_limit=100):
        if source_key is None and hasattr(self, 'train_key'):
            source_key = self.train_key
        source_data = self.data[source_key]
        data = {}
        train_x, train_y = source_data.ordered_dataloader.dataset.tensors[:2]
        train_x, train_y = train_x.cpu().numpy(), train_y.cpu().numpy()
        coords = train_y[:,0,1:]
        for key in self.location_index_dict:
            name = self.location_index_dict[key]
            if isinstance(name, str):
                if 'bus' in name: continue
                if combine_sensors:
                    if 'nuc' in name and 'b210' in name:
                        name = name[:-6]
                    elif 'cellsdr' in name:
                        name = name.replace('cell','cbrs')
            rss = train_x[:,key,0]
            valid_inds = rss != 0
            if valid_inds.sum() < required_limit: continue
            sensor_loc = train_x[valid_inds][0,key,1:3]
            sensor_type = train_x[valid_inds][0,key,4]
            rss = rss[valid_inds]
            if combine_sensors and use_db_for_combination:
                min_rss, max_rss = self.get_min_max_rss_from_key(key)
                rss = rss * (max_rss - min_rss) + min_rss
            few_coords = coords[valid_inds]
            if name not in data:
                data[name] = [few_coords, rss, sensor_loc, [key], sensor_type]
            else:
                data[name][3].append(key)
                data[name] = [
                    np.concatenate((data[name][0], few_coords)),
                    np.concatenate((data[name][1], rss)),
                    sensor_loc,
                    data[name][3],
                    sensor_type
                    ]
        for name in data:
            few_coords, rss, sensor_loc, keys, sensor_type = data[name]
            sorted_inds = np.lexsort(few_coords.T)
            few_coords = few_coords[sorted_inds]
            rss = rss[sorted_inds]
            if combine_sensors and use_db_for_combination:
                min_rss = min(self.get_min_max_rss_from_key(key)[0] for key in keys)
                max_rss = min(self.get_min_max_rss_from_key(key)[1] for key in keys)
                rss = (rss - min_rss) / (max_rss - min_rss)
            row_mask = np.append([True], np.any(np.diff(few_coords,axis=0),1))
            data[name][0] = few_coords[row_mask]
            data[name][1] = rss[row_mask]
        return data

    def plot_separation(self, train_key=None, test_key=None, labels=None, plot_rx=False, save_plot=False, data_file=None):
        if train_key is None:
            train_key = self.train_key
        if test_key is None:
            test_key = self.test_key if self.test_key is not None else self.test_keys[0]
        all_train_tx = []
        all_test_tx = []
        all_train_rx = []
        all_test_rx = []
        for entry in self.data[train_key].rx_vecs:
            for rx in entry:
                all_train_rx.append(rx[1:])
        for entry in self.data[test_key].rx_vecs:
            for rx in entry:
                all_test_rx.append(rx[1:])
        for entry in self.data[train_key].tx_vecs:
            for tx in entry:
                all_train_tx.append(tx)
        for entry in self.data[test_key].tx_vecs:
            for tx in entry:
                all_test_tx.append(tx)
        all_train_rx = np.array(all_train_rx)
        all_test_rx = np.array(all_test_rx)
        all_train_tx = np.array(all_train_tx)
        all_test_tx = np.array(all_test_tx)
        train_tx_label = (train_key if labels is None else labels[0])
        test_tx_label = (test_key if labels is None else labels[1])
        rx_label = 'Sensors'
        plt.scatter(all_train_tx[:,0], all_train_tx[:,1], c='tab:red', label=train_tx_label, alpha=0.7, marker='+')
        plt.scatter(all_test_tx[:,0], all_test_tx[:,1], c='tab:green', label=test_tx_label, alpha=0.7, marker='o')
        if plot_rx:
            plt.scatter(all_train_rx[:,0], all_train_rx[:,1], c='blue', label=rx_label, alpha=0.2, marker='^', s=5)
            plt.scatter(all_test_rx[:,0], all_test_rx[:,1], c='blue', alpha=0.2, marker='^', s=5)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if self.params.dataset_index == 6:
            plt.xlim(60,2290)
            plt.ylim(360,2230)
        plt.legend()
        plt.tight_layout()
        if data_file is None:
            data_file = 'dataset_scatter_scatter_%i.png' % (self.params.dataset_index + 1)
        if save_plot:
            plt.savefig(data_file)
            plt.cla()
        else:
            plt.show()

# Consolidate rx_tups according to tx in tx_locs, instead of one pair per row
# this assumes only 1 txer active at a time for a set of rx
# datalist contains a list of [tx_locs, rx_tups, tx_gains]
# precision is the difference (decimal places) in  UTM coords that distinguishes a different TX location
# precision also truncates (only) tx_locations to that level in the new_data_list
def consolidate_data(data_list, precision=2):
    #print(f"before data_list {data_list}")
    new_data_list_dict = {}
    for entry in data_list:
        tx_loc = np.round(entry[0],precision)
        if tx_loc.tobytes() not in new_data_list_dict:
            new_data_list_dict[tx_loc.tobytes()] = [tx_loc, entry[1], entry[2]]
        else:
            tmp = new_data_list_dict[tx_loc.tobytes()]
            #print(f"entry {entry} and tmp {tmp}")
            #print(f"tmp[1] {tmp[1]} and entry[1] {entry[1]}")
            new_data_list_dict[tx_loc.tobytes()] = [tmp[0], np.append(tmp[1],entry[1], axis=0), tmp[2]]
    #print(f"after data_list {list(new_data_list_dict.values())}")
    new_data_list = list(new_data_list_dict.values())
    return new_data_list


if __name__ == "__main__":
    #for ds  in [6,7,8]:
    for ds  in [8]:
      for split in ['random', 'grid10', 'radius100']:
        params = LocConfig(ds, data_split=split)
        rldataset = RSSLocDataset(params)
        rx_data = rldataset.data[rldataset.train_key].ordered_dataloader.dataset.tensors[0].cpu().numpy()
        tx_data = rldataset.data[rldataset.train_key].ordered_dataloader.dataset.tensors[1].cpu().numpy()
        rx = rx_data[rx_data[:,:,4] > 1]
        rx = rx[rx[:,0] > 0]
        all_rx = np.unique(rx[:,1:4], axis=0)
        rldataset.print_dataset_stats()
        embed()