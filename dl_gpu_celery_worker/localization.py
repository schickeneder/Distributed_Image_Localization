from collections import OrderedDict
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from dataset import RSSLocDataset
# from attacker import get_random_attack_vec, worst_case_attack

from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import Bounds
import re
import code
from itertools import product


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def calc_distances(txes, rxes):
    # distances = np.linalg.norm(end_points - start_points, axis=1)

    try:
        distances = np.sqrt((txes[:,0]-rxes[:,0])*(txes[:,0]-rxes[:,0]) + (txes[:,1]-rxes[:,1])*(txes[:,1]-rxes[:,1]))
        return distances
    except:
        pass

    try:
        distances = np.sqrt((txes[0] - rxes[0]) * (txes[0] - rxes[0]) + (txes[1] - rxes[1]) * (txes[1] - rxes[1]))
        return distances
    except:
        pass

    print(f"failed to calculate distances for coords {txes}, {rxes}")
    return None





class PhysLocalization():
    # runs physics-based MSE pathloss model(s)
    def __init__(
            self,
            rss_loc_dataset: RSSLocDataset,

    ):
        self.params = rss_loc_dataset.params
        self.rss_loc_dataset = rss_loc_dataset
        if self.params.include_elevation_map:  # not yet implemented
            pass
        self.device = self.rss_loc_dataset.params.device  # for CUDA, may or may not use..
        self.img_size = np.array([self.rss_loc_dataset.img_height(), self.rss_loc_dataset.img_width()])
        self.linear_PL = 0
        self.log_PL = 0
        self.per_node_dist_rss_array = [] # [[[dist,rss],[]..],[[dist,rss],[]]..]
        self.per_node_PL_array = [] # calculated PL across all tx-rx pairs for each TXing node
        self.dist_rss_array = []  # [[distance,rss]..]
        self.per_node_error_vector_array = []
        self.regular_error_vector_array = []
        self.vector_error_factor_array = []
        self.rss_dist_ratio = None
        self.linear_PL = None
        self.log_PL = None
        self.PL_exp = None

        self.calculate_pathloss()
        tmp = self.PL_exp
        self.calculate_per_node_pathloss() # have to be careful because this overwrites global PL_exp
        self.PL_exp = tmp # temporary workaround for per_node_pathloss overwriting
        self.calculate_per_node_error() # this calculates error vector arrays which will be needed by test
        self.calculate_error()

        #self.test_model()

    def pathloss_error_optimizer_function(self, pl_factor, option):
        # helper function for finding the optimal pathloss exponent (factor)
        error_array = self.calculate_error(pl_factor, option)
        return error_array.mean()

    def add_vector_error_array_offset(self,tx_locs, rx_locs, true_dist, rx_dist_est,index):
        # need to project the magnitude the error vector array onto the true distance vectors

        # get x, y dist components
        x_tmps = np.array(tx_locs[:, 0] - rx_locs[:, 0], dtype=float)
        y_tmps = np.array(tx_locs[:, 1] - rx_locs[:, 1], dtype=float)

        dist_vecs = np.column_stack((x_tmps, y_tmps))

        projected_error_magnitudes = np.sum(dist_vecs * np.array(self.per_node_error_vector_array[index]), axis=1) / true_dist

        # returns new rx_dist_est by adjusting based on error_vector_array offset
        return projected_error_magnitudes


    def calculate_pathloss(self):
        # Calculates self.log_PL and self.linear_PL as global pathloss values for use in estimations
        # here we don't need separate validation data, so all can be used

        # distances = calc_distances(self.params.start_points, self.params.end_points)

        # physloc.rss_loc_dataset.data (data is type Samples from dataset.py) with [None] being the full sample set

        # print(f"rx_vecs from self.rss_loc_dataset.data['0.2testsize_train'] {self.rss_loc_dataset.data['0.2testsize_train'].rx_vecs}")
        # print(f"tx_vecs from self.rss_loc_dataset.data['0.2testsize_train'] {self.rss_loc_dataset.data['0.2testsize_train'].tx_vecs}")
#        print(self.rss_)

        #self.rss_loc_dataset.data[None].rx_vecs[0][:, 1:3]  # just the lat/lon (y/x) coordinates

        tx_count = len(self.rss_loc_dataset.data[None].tx_vecs)

        for index in range(tx_count):
            txes = np.array(
                [self.rss_loc_dataset.data[None].tx_vecs[index]] * len(self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]),
                dtype=float)[:, 0]
            rxes = self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]

            rss_tmps = np.array(self.rss_loc_dataset.data[None].rx_vecs[index][:, 0:1], dtype=float)
            dist_tmps = np.array(calc_distances(txes, rxes))

            for dist_tmp, rss_tmp in zip(dist_tmps, rss_tmps):
                self.dist_rss_array.append([float(dist_tmp), float(rss_tmp)])
        # for tx_vec, rx_vec in zip(self.rss_loc_dataset.data[None].tx_vecs,self.rss_loc_dataset.data[None].rx_vecs):
        #     distances = np.linalg.norm(np.array(tx_vec*len(rx_vec), dtype=float) - np.array(rx_vec[:, 1:3],dtype=float), axis=1)
        #     rss = np.array(rx_vec[:, 0:1],dtype=float)
        #     distances_array.append(distances)
        #     rss_array.append(rss)

        # for a linear pathloss factor

        tmp_array = np.array(self.dist_rss_array)
        tmp_array = np.ma.masked_equal(tmp_array, 0)
        res = tmp_array[:, 0] / tmp_array[:, 1]

        self.rss_dist_ratio = res.mean()

        # linear regression
        # self.linear_PL = minimize_scalar(self.pathloss_error_optimizer_function, bounds=(0, 1000000), method='bounded', args="rss_dist_ratio").fun

        # code.interact(local=locals())

        #logarithmic regression
        res = minimize_scalar(self.pathloss_error_optimizer_function, bounds=(-10000, 10000), method='bounded', args="MMSE")
        print(f"result of PL_exp min {res}")
        self.PL_exp = res.x
        print(f"PL_exp is {self.PL_exp}")
        print(f"test result from error_optimizer_function {self.pathloss_error_optimizer_function(self.PL_exp, 'MMSE')}")




        # TODO implement a pointwise interpolated regression function; i.e. for each RSS value it looks up distance
        # in a lookup table or interpolates if no data exists for that range; empirically derived look-up table
        # call this one discrete pathloss

        # code.interact(local=locals())

    def calculate_per_node_pathloss(self):
        # Calculates pathloss per TX node by only solving for TX-RX pairs associated with that node
        # Per node gain can be calculated by subtracting global pathloss from this value

        # distances = calc_distances(self.params.start_points, self.params.end_points)

        # physloc.rss_loc_dataset.data (data is type Samples from dataset.py) with [None] being the full sample set

        # print(f"rx_vecs from self.rss_loc_dataset.data['0.2testsize_train'] {self.rss_loc_dataset.data['0.2testsize_train'].rx_vecs}")
        # print(f"tx_vecs from self.rss_loc_dataset.data['0.2testsize_train'] {self.rss_loc_dataset.data['0.2testsize_train'].tx_vecs}")
        #        print(self.rss_)

        #self.rss_loc_dataset.data[None].rx_vecs[0][:, 1:3]  # just the lat/lon (y/x) coordinates


        tx_count = len(self.rss_loc_dataset.data[None].tx_vecs)
        tmp_dist_rss_array = []
        error_array = []

        for index in range(tx_count):
            txes = np.array(
                [self.rss_loc_dataset.data[None].tx_vecs[index]] * len(
                    self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]),
                dtype=float)[:, 0]
            rxes = self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]

            rss_tmps = np.array(self.rss_loc_dataset.data[None].rx_vecs[index][:, 0:1], dtype=float)
            # should also store distance vectors so we can determine relative angle between tx-rx
            # it may be that error is more directional
            x_tmps = np.array(txes[:,0]-rxes[:,0], dtype=float)
            y_tmps = np.array(txes[:,1]-rxes[:,1], dtype=float)
            dist_tmps = np.array(calc_distances(txes, rxes))

            self.dist_rss_array = []
            for dist_tmp, rss_tmp, x_tmp, y_tmp in zip(dist_tmps, rss_tmps, x_tmps, y_tmps):
                self.dist_rss_array.append([float(dist_tmp), float(rss_tmp), float(x_tmp), float(y_tmp)])
                tmp_dist_rss_array.append([float(dist_tmp), float(rss_tmp), float(x_tmp), float(y_tmp)])

            # logarithmic regression - we won't bother with linear
            res = minimize_scalar(self.pathloss_error_optimizer_function, bounds=(-10000, 10000), method='bounded', args="MMSE")
            self.PL_exp = res.x

            self.per_node_PL_array.append(self.PL_exp)

            self.per_node_dist_rss_array.append(self.dist_rss_array) # [[[dist,rss,x,y],[]..],[[dist,rss,x,y],[]]..]




        self.dist_rss_array = tmp_dist_rss_array # setting dist_rss_array back to the full list of [[dist,rss],[],..]
        # print(f"error per node for per_node PL_exp is {error_array}")
        # print(f"per_node_dist_rss_array {self.per_node_dist_rss_array}")
        # print(f"PL_exp {self.PL_exp}")
        # print(f"per node PL array {self.per_node_PL_array}")









    def calculate_error(self, pl_factor=None, option="MMSE"):
    # calculates error due to pathloss factor estimate, used in optimization function and initialization
        if not pl_factor:
            pl_factor = self.PL_exp
        dist_array_est = 10 ** ((np.array(self.dist_rss_array)[:, 1]) / (10 * pl_factor))

        if option == "MMAE":
            error_array = np.abs(np.array(self.dist_rss_array)[:, 0] - dist_array_est)

        elif option == "MMSE":
            error_array = (np.array(self.dist_rss_array)[:, 0] - dist_array_est) ** 2

        else:
            return None

        error_array = error_array[np.isfinite(error_array)]  # get rid of any inf or NaN values

        return error_array

    def error_vector_optimization_function(self, average_error_vector, unit_vector_dist_array,index):
        vector_error_factor = np.dot(unit_vector_dist_array, average_error_vector)

        dist_array_error = (10 ** ((np.array(self.per_node_dist_rss_array[index])[:, 1]) /
                                    (10 * np.array(self.per_node_PL_array[index])))
                             - np.array(self.per_node_dist_rss_array[index])[:, 0]) - vector_error_factor

        return np.mean(np.abs(dist_array_error))



    def calculate_per_node_error(self):
        tx_count = len(self.per_node_PL_array)
        per_node_error_array = [] # each node uses its own PL_exp
        regular_error_array = [] # each node uses the same global PL_exp

        # print(f"dist_rss_array {np.array(self.dist_rss_array)[:, 1]}")
        # print(f"per_node_PL_array {self.per_node_PL_array[0]}")
        # print(f"per_node_dist_rss_array {np.array(self.per_node_dist_rss_array[0])}")

        for index in range(tx_count):
            tmp_error_vector_array = []
            dist_array_error = 10 ** ((np.array(self.per_node_dist_rss_array[index])[:, 1]) / (10 * np.array(self.per_node_PL_array[index]))) - np.array(
                self.per_node_dist_rss_array[index])[:, 0]
            # compute unit vector then multiply by error distance. Note: rows are [dist,rss,x,y]
            unit_vector_dist_array = np.column_stack((np.array(self.per_node_dist_rss_array[index])[:, 2] /
                                                  np.array(self.per_node_dist_rss_array[index])[:, 0],
                                                  np.array(self.per_node_dist_rss_array[index])[:, 3] /
                                                  np.array(self.per_node_dist_rss_array[index])[:, 0]))

            vector_array_error = unit_vector_dist_array * dist_array_error[:, np.newaxis]
            average_error_vector = np.mean(vector_array_error,axis=0) # average error vector for this index
            # if we offset

            res = minimize(self.error_vector_optimization_function, [100,100], args=(unit_vector_dist_array,index),method='trust-constr')
            # print(res)


            # without vector_error_factor

            dist_array_error0 = (10 ** ((np.array(self.per_node_dist_rss_array[index])[:, 1]) /
                                       (10 * np.array(self.per_node_PL_array[index])))
                                 - np.array(self.per_node_dist_rss_array[index])[:, 0])

            # minimize vector_error_factor --- this one seems to perform best
            vector_error_factor = np.dot(unit_vector_dist_array, res.x)

            dist_array_error = (10 ** ((np.array(self.per_node_dist_rss_array[index])[:, 1]) /
                                       (10 * np.array(self.per_node_PL_array[index])))
                                 - np.array(self.per_node_dist_rss_array[index])[:, 0]) - vector_error_factor

            self.vector_error_factor_array.append(vector_error_factor)

            # calculate distance offset in the direction of these nodes

            # analytically computed vector_error_factor
            vector_error_factor = np.dot(unit_vector_dist_array, average_error_vector)


            dist_array_error2 = (10 ** ((np.array(self.per_node_dist_rss_array[index])[:, 1]) /
                                       (10 * np.array(self.per_node_PL_array[index])))
                                 - np.array(self.per_node_dist_rss_array[index])[:, 0]) - vector_error_factor
            # print(f" dist_array_error sum = {np.sum(dist_array_error)}, dist_array_error2 sum = {np.sum(dist_array_error2)}")
            # print(f" dist_array_error average = {np.mean(dist_array_error)}, dist_array_error2 average = {np.mean(dist_array_error2)}")
            # print(f" dist_array_error average abs = {np.mean(np.abs(dist_array_error))}, dist_array_error2 average abs = {np.mean(np.abs(dist_array_error2))}")

            # print(f"average_error_vector {average_error_vector}")

            # print(f"average_error_vector {average_error_vector} and minimized vector_factor {res.x}")
            # print(f" avg abs without factor, optim, analy = {np.mean(np.abs(dist_array_error0 )):0f} {np.mean(np.abs(dist_array_error)):0f} {np.mean(np.abs(dist_array_error2)):0f}")

            # continue
            #
            # print("In calculate_per_node_error")
            # code.interact(local=locals())


            # add or subtract mean vector array error from x-y components

            # print(f"per node sum vector error {np.sum(vector_array_error,axis=0)}")
            # print(f"per node mean vector error {np.mean(vector_array_error,axis=0)}")

            per_node_error_array.append(np.mean(dist_array_error))
            self.per_node_error_vector_array.append(np.mean(vector_array_error,axis=0))

            dist_array_error = 10 ** ((np.array(self.per_node_dist_rss_array[index])[:, 1]) / (10 * self.PL_exp)) - np.array(
                self.per_node_dist_rss_array[index])[:, 0]
            vector_array_error = np.column_stack((np.array(self.per_node_dist_rss_array[index])[:, 2] /
                                                  np.array(self.per_node_dist_rss_array[index])[:, 0] * dist_array_error,
                                                  np.array(self.per_node_dist_rss_array[index])[:, 3] /
                                                  np.array(self.per_node_dist_rss_array[index])[:, 0] * dist_array_error))

            # print(f"reg sum vector error {np.sum(vector_array_error,axis=0)}")
            # print(f"reg mean vector error {np.mean(vector_array_error,axis=0)}")

            regular_error_array.append(np.mean(dist_array_error))
            self.regular_error_vector_array.append(np.mean(vector_array_error,axis=0))
            # but we also want to calculate the net error vector for each tx
            # derive unit vector from x, y of dist_rss_array [dist, rss, x, y] as x/dist y/dist

        # print(f"per_node_PL_exp error {per_node_error_array}")
        # print(f"regular_PL_exp error {regular_error_array}")
        # print(f"mean, median, min, max per node error {np.mean(per_node_error_array)},"
        #       f" {np.median(per_node_error_array)}, {np.min(per_node_error_array)}, {np.max(per_node_error_array)}")
        # print(f"mean, median, min, max regular error {np.mean(regular_error_array)},"
        #       f" {np.median(regular_error_array)}, {np.min(regular_error_array)}, {np.max(regular_error_array)}")
        #
        # print(f"mean, median, min, max per node vector error {np.mean(self.per_node_error_vector_array,axis=0)},"
        #       f" {np.median(self.per_node_error_vector_array,axis=0)}, {np.min(self.per_node_error_vector_array,axis=0)}, "
        #       f"{np.max(self.per_node_error_vector_array,axis=0)}")
        # print(f"mean, median, min, max regular vector error {np.mean(self.regular_error_vector_array,axis=0)},"
        #       f" {np.median(self.regular_error_vector_array,axis=0)}, {np.min(self.regular_error_vector_array,axis=0)}, "
        #       f"{np.max(self.regular_error_vector_array,axis=0)}")
        return

    # tests pathloss model against receivers
    def test_model(self, option="log10"):
        estimate_error_list = []
        max_x, min_x, max_y, min_y = (self.rss_loc_dataset.max_x, self.rss_loc_dataset.min_x,
                                      self.rss_loc_dataset.max_y, self.rss_loc_dataset.min_y)
        # print(f"image size is {self.img_size} and meter scale is {self.params.meter_scale}")
        x_grids = np.arange(min_x, max_x, self.params.meter_scale)
        y_grids = np.arange(min_y, max_y, self.params.meter_scale)

        grid_test_array = np.array([(x, y) for x,y in product(x_grids, y_grids)]) # reversing order because got swapped

        tx_count = len(self.rss_loc_dataset.data[None].tx_vecs)

        for index in range(tx_count):  # go through each tx and set of rxes
            txes = np.array(
                [self.rss_loc_dataset.data[None].tx_vecs[index]] * len(
                    self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]),
                dtype=float)[:, 0]
            rxes = self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]
            rxrss = self.rss_loc_dataset.data[None].rx_vecs[index][:, 0]  # normalized rss value

            if option == "rss_dist_ratio":
                rx_dist_est = np.array(rxrss) * self.rss_dist_ratio + self.linear_PL # probably should remove this opt.
            else:  # default
                rx_dist_est = 10 ** (( np.array(rxrss)) / (10 * self.PL_exp))
                        #10 ** ( np.array(rxrss) / (10 * self.PL_exp)))
                        #np.log10(np.array(rxrss)) + self.PL_exp)  # distance based on rss via model, assumes tx pwr = 0

            # test the distance error for each "pixel" over the area to produce an estimate

            sum_of_dists_list = []
            for coords in grid_test_array:
                tx_loc = np.array([coords]*len(rxes)) # expand a single grid location to quickly test against all rxes
                true_dist = calc_distances(tx_loc,rxes) # distances between coords and each point in the test grid

                res = abs(true_dist - rx_dist_est) # distance difference between true and estimated
                sum_of_dists_list.append(np.sum(res))
                # min_dist = np.min(res) # find the smallest distance
                # min_dist_index = np.argmin(res)
                # print(f"sum of distances for {coords} {np.sum(res)}")
                # print(f"min distance: {min_dist} at index {min_dist_index}")
            res_array = np.array(sum_of_dists_list)
            min_dist = np.min(res_array)  # the pixel whose sum of distances to RXes is minimum
            min_dist_index = np.argmin(res_array)  # the corresponding index to link to the TX
            # print(f"for tx {txes[0]} the min distance est is {min_dist} at location {grid_test_array[min_dist_index]}")
            error = calc_distances(txes[0], grid_test_array[min_dist_index])
            estimate_error_list.append(error)
            # print(f"the error is {error}")
            # code.interact(local=locals())
            #
            # for x in x_grids:
            #     for y in y_grids:

        # TODO: need to finish this, and will need to optimize otherwise it will take forever

        return np.array(estimate_error_list).mean()

    # tests pathloss model against receivers
    def test_model_per_node_PL(self, option="log10"):
        estimate_error_list = []
        RMS_error_factor_array = []
        max_x, min_x, max_y, min_y = (self.rss_loc_dataset.max_x, self.rss_loc_dataset.min_x,
                                      self.rss_loc_dataset.max_y, self.rss_loc_dataset.min_y)
        # print(f"image size is {self.img_size} and meter scale is {self.params.meter_scale}")
        x_grids = np.arange(min_x, max_x, self.params.meter_scale)
        y_grids = np.arange(min_y, max_y, self.params.meter_scale)

        grid_test_array = np.array([(x, y) for x,y in product(x_grids, y_grids)])

        tx_count = len(self.rss_loc_dataset.data[None].tx_vecs)

        for index in range(tx_count):  # go through each tx and set of rxes
            RMS_error_factor_array.append(np.sqrt(np.mean(np.square(np.array(self.vector_error_factor_array[index])))))

            txes = np.array(
                [self.rss_loc_dataset.data[None].tx_vecs[index]] * len(
                    self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]),
                dtype=float)[:, 0]
            rxes = self.rss_loc_dataset.data[None].rx_vecs[index][:, 1:3]
            rxrss = self.rss_loc_dataset.data[None].rx_vecs[index][:, 0]  # normalized rss value

            if option == "rss_dist_ratio":
                rx_dist_est = np.array(rxrss) * self.rss_dist_ratio + self.linear_PL
            else:  # default
                rx_dist_est = (10 ** (( np.array(rxrss)) / (10 * self.per_node_PL_array[index]))
                               - self.vector_error_factor_array[index])

            # test the distance error for each "pixel" over the area to produce an estimate

            sum_of_dists_list = []
            sum_of_dists_list_vector_adjusted = []
            for coords in grid_test_array:
                tx_loc = np.array([coords]*len(rxes)) # expand a single grid location to quickly test against all rxes
                # TODO instead of calc_distances, how about just using x,y components; would need to have component
                # rx_dist_est then..
                true_dist = calc_distances(tx_loc,rxes) # distances between coords and each point in the test grid

                # including projected error offsets
                projected_error_mags = self.add_vector_error_array_offset(tx_loc, rxes, true_dist,rx_dist_est,index)



                # print(f"rx_dist_est, projected_error_mags {rx_dist_est[:5],projected_error_mags[:5]}")
                # rx_dist_est = rx_dist_est + projected_error_mags

                res = abs(true_dist - rx_dist_est) # distance difference between true and estimated
                sum_of_dists_list.append(np.sum(res))

                res = abs(true_dist - (rx_dist_est + projected_error_mags))
                sum_of_dists_list_vector_adjusted.append(np.sum(res))


                # min_dist = np.min(res) # find the smallest distance
                # min_dist_index = np.argmin(res)
                # print(f"sum of distances for {coords} {np.sum(res)}")
                # print(f"min distance: {min_dist} at index {min_dist_index}")

            # print("In test_model_per_node_PL")
            # code.interact(local=locals())

            res_array = np.array(sum_of_dists_list)
            min_dist = np.min(res_array)  # the pixel whose sum of distances to RXes is minimum
            min_dist_index = np.argmin(res_array)  # the corresponding index to link to the TX
            # print(f"for tx {txes[0]} the min distance est is {min_dist} at location {grid_test_array[min_dist_index]}")
            error = calc_distances(txes[0], grid_test_array[min_dist_index]) # txes[0] because they're all the same val
            estimate_error_list.append(error)
            # print(f"the error is {error}")
            # code.interact(local=locals())
            #
            # for x in x_grids:
            #     for y in y_grids:

        # TODO: need to finish this, and will need to optimize otherwise it will take forever
        # returns average of RMS of applied error factor offsets and the average estimation error for that area
        return np.array(RMS_error_factor_array).mean(),np.array(estimate_error_list).mean()

    def get_data_distribution_stats(self):
        witness_events_count_list = []
        max_x, min_x, max_y, min_y = (self.rss_loc_dataset.max_x, self.rss_loc_dataset.min_x,
                                      self.rss_loc_dataset.max_y, self.rss_loc_dataset.min_y)
        # print(f"image size is {self.img_size} and meter scale is {self.params.meter_scale}")
        x_grids = np.arange(min_x, max_x, self.params.meter_scale)
        y_grids = np.arange(min_y, max_y, self.params.meter_scale)

        grid_test_array = np.array([(x, y) for x,y in product(x_grids, y_grids)]) # reversing order because got swapped

        grid_counts = np.zeros((len(x_grids), len(y_grids)), dtype=int)

        tx_count = len(self.rss_loc_dataset.data[None].tx_vecs)

        for index in range(tx_count):  # go through each tx and set of rxes
            num_witness_events = len(self.rss_loc_dataset.data[None].rx_vecs[index])
            witness_events_count_list.append(num_witness_events)

        for val in self.rss_loc_dataset.data[None].tx_vecs:
            x,y = val[0]
            x_index = np.searchsorted(x_grids, x, side='right') - 1
            y_index = np.searchsorted(y_grids, y, side='right') - 1
            if 0 <= x_index < len(x_grids) and 0 <= y_index < len(y_grids):
                grid_counts[x_index, y_index] += 1

        # print(f"witness events: {witness_events_count_list}")
        # print(f"grid counts: {grid_counts}")


        flat_counts = grid_counts.flatten()
        hist, bin_edges = np.histogram(flat_counts, bins=np.arange(flat_counts.max() + 2))

        # returns witness events count list and histogram for grid counts
        print(witness_events_count_list, hist, bin_edges)
        return witness_events_count_list, hist, bin_edges




class DLLocalization():
    # runs DL Localization models
    def __init__(
            self,
            rss_loc_dataset: RSSLocDataset,
            loss_object=SlicedEarthMoversDistance(num_projections=200, scaling=10),
            lr=1e-4
    ):
        self.params = rss_loc_dataset.params
        self.rss_loc_dataset = rss_loc_dataset
        if self.params.include_elevation_map:
            self.rss_loc_dataset.make_elevation_tensors()
        self.device = self.rss_loc_dataset.params.device
        self.img_size = np.array([self.rss_loc_dataset.img_height(), self.rss_loc_dataset.img_width()])
        self.loss_func = loss_object

        self.build_model(self.params.arch, lr=lr)

    def build_model(self, arch='unet_ensemble', output_channel=1, lr=1e-4):
        """
        Initialize the UNet Ensemble model and setup optimizer
        """
        self.learn_locs = False
        self.calc_distance = False
        n_channels = 1 if not self.params.include_elevation_map else 2
        arch_without_number = ''.join(i for i in arch if not i.isdigit())
        channel_scale = get_trailing_number(arch)
        channel_scale = channel_scale if channel_scale is not None else 32
        depth = 4
        if arch_without_number == 'unet':
            net_model = EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=1, channel_scale=channel_scale, input_resolution=self.params.meter_scale, depth=depth)#, elevation_map=self.rss_loc_dataset.elevation_tensors[0])
        elif arch_without_number == 'unet_tiny':
            net_model = EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=1, channel_scale=8, input_resolution=self.params.meter_scale, depth=3)
        elif arch_without_number == 'unet_ensemble':
            net_model = EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=5, channel_scale=channel_scale, input_resolution=self.params.meter_scale, depth=depth)
        elif arch == 'mlp':
            net_model = MLPLocalization(self.rss_loc_dataset.max_num_rx)
        else:
            raise NotImplementedError
        self.model = net_model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = optimizer

    def predict_img_many(self, x_vecs):
        if isinstance(self.loss_func, CoMLoss):
            pred_vecs = self.model.com_predict(x_vecs)
        else:
            pred_vecs = self.model.predict(x_vecs)
        return pred_vecs[:, :2]

    def set_rss_tensor(self):
        if not hasattr(self, 'rss_tensor'):
            train_dataloader = self.rss_loc_dataset.data[self.rss_loc_dataset.train_key].ordered_dataloader
            rss_inds = train_dataloader.dataset.tensors[0][:, :, 0] != 0
            all_rss = train_dataloader.dataset.tensors[0][:, :, 0][rss_inds]
            self.rss_tensor = torch.quantile(all_rss, torch.tensor([0.1, 0.9], device=self.device))

    def test(self, test_key=None, dataloader=None, num_power_repeats=1, save_images=False, apply_wc_attack=False):
        """Evaluate model on the given dataloader dataset or test keys

        Args:
            dataloader      torch.DataLoader -- data to evaluate
            y_vecs          list<np.array> -- ground truth for locations
            num_power_repeats  int -- number of times to repeat testset, if assigning random power each iteration, to get avg performance
        return:
            total_loss      float -- loss from testset
            best_results    dict -- results from best setting of thresh and suppression_size
            min_fn          float -- misdetection rate
            min_fp          float -- false alarm rate
        """
        self.model.eval()
        all_x_images = []
        all_y_images = []
        all_pred_images = []
        all_pred_vecs = []
        all_error_vecs = []
        if dataloader is None or test_key is not None:
            dataloader = self.rss_loc_dataset.data[test_key].ordered_dataloader
        for _ in range(num_power_repeats):
            repeat_pred_vecs = []
            repeat_error_vecs = []
            repeat_x_images = []
            repeat_y_images = []
            repeat_pred_images = []
            for t, sample in enumerate(dataloader):
                x_vecs = sample[0].to(self.device)
                y_vecs = sample[1].to(self.device)

                if save_images:
                    pred_imgs, x_img, y_img = self.model((x_vecs, y_vecs))
                    if isinstance(self.loss_func, CoMLoss):
                        pred_vecs = self.model.com_predict(pred_imgs, input_is_pred_img=True)
                    else:
                        pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)
                    # pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)
                    repeat_x_images.append(x_img.detach().cpu().numpy())
                    repeat_y_images.append(y_img.detach().cpu().numpy())
                    repeat_pred_images.append(pred_imgs.detach().cpu().numpy())
                else:
                    if isinstance(self.loss_func, CoMLoss):
                        pred_vecs = self.model.com_predict(x_vecs)
                    else:
                        pred_vecs = self.model.predict(x_vecs)
                repeat_pred_vecs.append(pred_vecs.detach().cpu().numpy())
                repeat_error_vecs.append(torch.linalg.norm(pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())
            if save_images:
                all_x_images.append(np.concatenate(repeat_x_images))
                all_y_images.append(np.concatenate(repeat_y_images))
                all_pred_images.append(np.concatenate(repeat_pred_images))
            all_pred_vecs.append(np.concatenate(repeat_pred_vecs))
            all_error_vecs.append(np.concatenate(repeat_error_vecs))
        all_pred_vecs = np.array(all_pred_vecs)
        all_error_vecs = np.array(all_error_vecs) * self.params.meter_scale

        results = {'preds': all_pred_vecs, 'error': all_error_vecs}
        if save_images:
            results['x_imgs'] = np.array(all_x_images)
            results['y_imgs'] = np.array(all_y_images)
            results['pred_imgs'] = np.array(all_pred_images)
        return results

    def adv_train_step(self, x_vec, x_img, y_vec, epsilon=0.5):
        self.set_rss_tensor()
        rand_select = np.random.random()
        if rand_select < 0.5:
            return
        grad = x_img.grad.data.clone()
        adv_x = get_random_attack_vec(x_vec, grad, self.rss_tensor[0].item(), self.rss_tensor[1].item(), epsilon)
        pred_img, x_img, y_img = self.model((adv_x, y_vec))
        if isinstance(self.loss_func, nn.MSELoss):
            loss = self.loss_func(pred_img, y_img)
        else:
            loss = self.loss_func(pred_img, y_img, y_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_worst_attack(self, x_vecs, y_vecs):
        preds = worst_case_attack(x_vecs, y_vecs, self, self.img_size)


    def train_model(self, num_epochs, train_data_key=None, test_data_keys=None, verbose=True, save_model_file='', load_model=True, load_model_file='', restart_optimizer=False):
        """Train model using train_data_dict, evaluation on test_data_dict

        Args:
            num_epochs          num epochs for training
            train_data_keys     str or List[str] of keys in rss_loc_dataset.data to train model on
            test_data_keys      str or List[str] of keys in rss_loc_dataset.data to evaluate model on
            save_model_file     str -- filename to save torch model
            load_model          bool -- if true, load the model in load_model_file or save_model_file before resuming training
            load_model_file     str -- filename to load torch model from
            restart_optimizer   bool -- if loading model, restart the optimizer rather than resume from save file
        """
        if train_data_key is None:
            train_data_key = self.rss_loc_dataset.train_key
        if test_data_keys is None:
            if len(self.rss_loc_dataset.test_keys) > 0:
                test_data_keys = self.rss_loc_dataset.test_keys
            else:
                test_data_keys = [self.rss_loc_dataset.test_key]
        train_dataloader = self.rss_loc_dataset.data[train_data_key].dataloader
        rss_inds = train_dataloader.dataset.tensors[0][:, :, 0] != 0
        all_rss = train_dataloader.dataset.tensors[0][:, :, 0][rss_inds]
        self.adv_rss_vec = np.quantile(all_rss.cpu(), [0.1, 0.9])
        test_errors = {key: [] for key in test_data_keys}
        train_loss_arr = np.zeros(num_epochs)
        epoch = 0
        best_epoch = 0

        if len(load_model_file) == 0:
            load_model_file = save_model_file
        for model_ext in ['model_train_val.']:
            load_model_file_ext = load_model_file.replace('model.', model_ext)
            if load_model and os.path.exists(load_model_file_ext):
                try:
                    checkpoint = torch.load(load_model_file_ext)
                    self.load_model(load_model_file_ext)
                    print('Loading model from %s' % load_model_file_ext)
                    if not restart_optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        epoch = checkpoint['epoch']
                        best_epoch = checkpoint['epoch']
                    break
                except:
                    pass

        saved_rss = train_dataloader.dataset.tensors[0][:, :, 0].clone()
        while epoch < num_epochs and epoch - best_epoch <= self.params.better_epoch_limit:
            self.model.train()
            epoch_loss = 0
            for t, sample in enumerate(train_dataloader):
                X_vec = sample[0].to(self.device)
                y_vec = sample[1].to(self.device)
                pred_img, x_img, y_img = self.model((X_vec, y_vec))
                if isinstance(self.loss_func, nn.MSELoss):
                    loss = self.loss_func(pred_img, y_img)
                else:
                    loss = self.loss_func(pred_img, y_img, y_vec)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if self.params.adv_train:
                    self.adv_train_step(X_vec, x_img, y_vec)

            self.model.eval()
            if verbose:
                endchar = '\n'
            else:
                endchar = '\r'
            if epoch % 10 == 0:
                res = self.test(dataloader=train_dataloader)
                res_max = res['preds'][0, :, 2]
                center_point = torch.tensor([1100, 1200]).to(self.device)
                # tr_radial_dist = torch.linalg.norm(train_dataloader.dataset.tensors[1].squeeze()[:,1:3] - center_point, axis=1)
                # plt.scatter(res_max, tr_radial_dist.cpu().detach())
                # plt.show()
                correlation = spearmanr(res['error'][0], res['preds'][0, :, 2])[0]
                train_err = res['error'].mean()
                quants = np.quantile(res['error'], [0.25, 0.5, 0.75, 1])
                print('Ep%i Train Loss:%.2e  Tr Mean: %.1f    25/50/75/100%% %.1f %.1f %.1f %.1f %.2f ' % ( epoch+1, epoch_loss, train_err, quants[0], quants[1], quants[2], quants[3], correlation), end=endchar)

            epoch_loss = epoch_loss / (t + 1)
            train_loss_arr[epoch] = epoch_loss
            result_string = 'Ep%i Tr%.2e  ' % (epoch + 1, epoch_loss)
            should_print = False
            for test_key in test_data_keys:
                res = self.test(test_key)
                correlation = spearmanr(res['error'][0], res['preds'][0, :, 2])[0]
                test_err = res['error'].mean()
                test_errors[test_key].append(test_err)

                quants = np.quantile(res['error'], [0.25, 0.5, 0.75, 1])
                result_string += '%s: Mean:%.1f  25/50/75/100%% %.1f %.1f %.1f %.1f %.2f ' % (test_key, test_err, quants[0], quants[1], quants[2], quants[3], correlation)

                if test_err == min(test_errors[test_key]):
                    if 'train_val' in test_key:
                        save_string = 'train_val'
                        best_epoch = epoch
                    elif 'test' in test_key:
                        save_string = 'test'
                    else:
                        continue
                    if len(save_model_file) > 0:
                        torch.save({
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'epoch': epoch,
                                    }, save_model_file.replace('model.', 'model_%s.' % save_string))
            print(result_string, end=endchar)# if should_print else '\r')
            epoch += 1
            if epoch - best_epoch > self.params.better_epoch_limit and best_epoch != 0:
                break
        return train_loss_arr, test_errors

    def load_model(self, model_path, device=None):
        checkpoint = torch.load(model_path, map_location=device)
        if 'down1.maxpool_conv.1.double_conv.0.weight' in checkpoint['model_state_dict']:
            new_model_checkpoint = OrderedDict()
            for key in checkpoint['model_state_dict']:
                layer_type = key.split('.')[0]
                if 'up' in layer_type or 'down' in layer_type:
                    layer_number = int(layer_type[-1])
                    new_key = key.replace(layer_type, layer_type[:-1] + ('s.%i' % (layer_number - 1)))
                    new_model_checkpoint[new_key] = checkpoint['model_state_dict'][key]
                else:
                    new_model_checkpoint[key] = checkpoint['model_state_dict'][key]
            self.model.load_state_dict(new_model_checkpoint, strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.params.device)
