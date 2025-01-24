
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import re
import csv
import numpy as np
from city_elevation_processor import test_and_plot_selection, get_square_corners, read_exact_elevation_from_zip_file
from results_processor import compare_denylist_to_normal
import glob
import os
import pickle
from datetime import datetime
from collections import Counter
import code


def plot_and_save_samples_vs_error(merged_file = 'cities_error_and_counts.csv'):#'20240912_deny_list_merged_output.csv' ):
    # Read data from CSV file
    #data = pd.read_csv('deny_list_merged_output.csv')
    data = pd.read_csv(merged_file,encoding="ISO-8859-1")

    # Extracting the first column as y values and the second column as x values
    y = data.iloc[:, 3] # samples_count
    x = data.iloc[:, 2] # error

    plt.scatter(x, y, c='blue', alpha=0.2, edgecolors='black')

    # Add titles and labels
    plt.title(f'Mean Error vs Sample Quantity')
    plt.ylabel('Number of Samples')
    plt.xlabel('Mean Error (m)')


    filename = f"samples_vs_error_plot_{len(x)}cities_denylist.png"
    plt.savefig(filename, dpi=300)
    # Show plot
    plt.show()

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(x, y)

    print(f'Pearson correlation coefficient: {correlation_coefficient}')
    print(f'P-value: {p_value}')
    print(f"cities count: {len(x)}")
    print(f"Mean samples_count {np.mean(y)} and error {np.mean(x)} ")

def plot_and_save_samples_vs_error2(merged_file = 'cities_error_and_counts.csv'):#'20240912_deny_list_merged_output.csv' ):
    # Read data from CSV file
    #data = pd.read_csv('deny_list_merged_output.csv')
    data = pd.read_csv(merged_file,encoding="ISO-8859-1")

    # Extracting the first column as y values and the second column as x values
    y = data.iloc[:, 3] # samples_count
    x = data.iloc[:, 2] # error

    plt.scatter(x, y, c='blue', alpha=0.2, edgecolors='black')

    # Add titles and labels
    plt.title(f'Mean Error vs Sample Quantity')
    plt.ylabel('Number of Samples')
    plt.xlabel('Mean Error (m)')


    filename = f"samples_vs_error_plot_{len(x)}cities_denylist.png"
    plt.savefig(filename, dpi=300)
    # Show plot
    plt.show()

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(x, y)

    print(f'Pearson correlation coefficient: {correlation_coefficient}')
    print(f'P-value: {p_value}')
    print(f"cities count: {len(x)}")
    print(f"Mean samples_count {np.mean(y)} and error {np.mean(x)} ")

def plot_denylist_and_error():
    # plots the error difference between normal->denylist against the ratio of denylist nodes to normal nodes

    results_dict = compare_denylist_to_normal()

    x_vals = []
    y_vals = []

    max_ratio = 1
    max_geonameid = ''
    for geonameid in results_dict:
        if results_dict[geonameid]['valid'] == 2:
            try:
                x1 = results_dict[geonameid]['normal_error']
                x2 = results_dict[geonameid]['denylist_error']
                error_diff = x1 - x2 # more positive is more improvement
                y1 = results_dict[geonameid]['normal_sample_counts']
                y2 = results_dict[geonameid]['denylist_sample_counts']
                counts_ratio =  float(y1)/float(y2)
                if y1 > y2: # we only care about sites where denylist was actually used..
                    x_vals.append(error_diff)
                    y_vals.append(counts_ratio)
                    # print(f"normal, denylist error {x1},{x2} counts {y1,y2}")
                    if counts_ratio > 1:
                        print(geonameid,results_dict[geonameid]['name'],counts_ratio, error_diff, y1, y2)
            except Exception as e:
                print(f"Error processing values {e}")

    plt.scatter(x_vals, y_vals, c='blue', alpha=0.2, edgecolors='black')

    # Add titles and labels
    plt.title(f'Error δ vs Denylist Ratio')
    plt.ylabel('Denylist Ratio')
    plt.xlabel('Mean Error δ (m)')


    filename = f"Error diff vs denylist ratio.png"
    plt.savefig(filename, dpi=300)
    # Show plot
    plt.show()

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(x_vals, y_vals)

    print(f'Pearson correlation coefficient: {correlation_coefficient}')
    print(f'P-value: {p_value}')
    print(f"cities count: {len(x_vals)}")
    print(f"Mean samples_count {np.mean(y_vals)} and error {np.mean(x_vals)} ")

# Fairland,US,39.07622,-76.95775,44.56766372846624 vs 2024_09_11-23_02_46-results.txt,Adeje__ES8.csv,1434.1417,4511
# can take city_elev_stdev.csv or city_elev_stdev_exact.csv with more precise selection area
def save_elevationstdev_vs_error(elev_stdev_file = 'city_elev_stdev_exact.csv',
                                 error_file = 'cities_error_and_counts.csv',
                                 outfile = '20240925_normal_error_vs_elev_stdev_exact.csv'):
    # reads the elevation file and outputs a combined outfile

    elev_stdev_data = pd.read_csv(elev_stdev_file, encoding="ISO-8859-1")
    error_data = pd.read_csv(error_file, encoding="ISO-8859-1")

    with open(outfile, 'w',newline='') as csvfile:
        print(f"opened {outfile} for writing")
        writer = csv.writer(csvfile)
        header = ["city_id","error","stdev"]
        writer.writerow(header)
        for index,row in error_data.iterrows():
            # print(f"index {index} row{row} city_id:{row['city_id']}")
            try:
                city_id = int(row['city_id'].split('_')[0].strip('"'))
                # print(city_id)
            except Exception as e:
                print(f"Failed to retrieve city_id for {row['city_id']} with error {e}")
                continue
            # print(city_id)
            for index2, row2 in elev_stdev_data.iterrows():
                try:
                    city_id2 = int(row2['geonameid'])
                    # print(city_id,city_id2)
                    if city_id == city_id2:
                        # print(city_id, city_id2)
                        # print(f"{city_id},{row['error']},{row2['stdev_elev']}")
                        try:
                            row_towrite = [city_id,row['error'],row2['stdev_elev']]
                            # print(row_towrite)
                            writer.writerow(row_towrite)
                        except Exception as e:
                            print(f"encountered an error {e} writing row {row_towrite}")

                except Exception as e:
                    print(f"encountered an error {e} in save_elevationstdev_vs_error")
                    continue

# plot error vs elevation stdev, file format is city_id,error,stdev
def plot_elevationstdev_vs_error(input_file = '20240912_error_vs_elev_stdev_exact.csv'):
    # Read data from CSV file
    data = pd.read_csv(input_file)

    # Extracting the first column as y values and the second column as x values
    y = data.iloc[:, 2] # elev stdev
    x = data.iloc[:, 1] # error

    plt.scatter(x, y, c='blue', alpha=0.2, edgecolors='black')

    # Add titles and labels
    plt.title(f'Mean Error vs Elevation σ')
    plt.ylabel('Elevation σ² (m)')
    plt.xlabel('Mean Error (m)')


    filename = f"error_vs_elev_stdev_plot_{len(x)}cities_denylist.png"
    plt.savefig(filename, dpi=300)
    # Show plot
    plt.show()

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(x, y)

    print(f'Pearson correlation coefficient: {correlation_coefficient}')
    print(f'P-value: {p_value}')
    print(f"cities count: {len(x)}")


def gps_to_array_coords(latitudes, longitudes, bottom_left, top_right, m, n):
    """
    Map GPS coordinates to an m x n array coordinate system.

    Parameters:
    latitudes (list of floats): List of latitude values to be mapped.
    longitudes (list of floats): List of longitude values to be mapped.
    bottom_left (tuple): Bottom-left GPS coordinate (latitude, longitude) of the bounding box.
    top_right (tuple): Top-right GPS coordinate (latitude, longitude) of the bounding box.
    m (int): Number of rows in the array.
    n (int): Number of columns in the array.

    Returns:
    list of tuples: Mapped (row, col) indices in the m x n array.
    """

    x_vals = []
    y_vals = []
    count = 0

    # Check that both lists have the same length
    if len(latitudes) != len(longitudes):
        raise ValueError("Latitudes and longitudes lists must have the same length.")

    # Unpack bounding box coordinates
    lat_min, lon_min = bottom_left
    lat_max, lon_max = top_right

    # Latitude and longitude ranges
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    print(f"m n lat_range lon_range m/n latr/lonr: {m}, {n}, {lat_range}, {lon_range}, {m/n}, {lat_range/lon_range}")


    for lat,lon in zip(latitudes,longitudes):
        if lat < lat_min or lat > lat_max or lon < lon_min or lon > lon_max:
            #print(f"({lat},{lon} out of range for bounds {bottom_left} and {top_right}")
            count += 1
            continue
        y = int((lat_max - lat) / lat_range * (m - 1))
        x = int((lon - lon_min) / lon_range * (n - 1))
        x_vals.append(x)
        y_vals.append(y)
    if count > 0:
        print(f"WARNING: {count}/{len(latitudes)} points fell outside of lat/lon bounds")
    return x_vals, y_vals


def plot_node_locations_on_elevation(geonameid = 5520993, denylist_filter = False):
    # [4791259, 'Virginia Beach', 689.3279]

    denylist = []
    count = 0
    if denylist_filter:
        with open('deny_lat_list.csv', 'r') as f:
            denylist = set(f.readline().strip().split(','))



    data_directory = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240921_15000cities_normal\generated"
    filepath = glob.glob(os.path.join(data_directory, str(geonameid) + '*'))[0]
    # print(filepath)
    lats = []
    lons = []
    with open(filepath,'r') as input_data_file:
        data = input_data_file.readlines()
    for line in data[1:]: # skip header line
        # print(line)
        lat = line.split(',')[1]
        lon = line.split(',')[2]

        if denylist_filter:
            # print(lat)
            if lat in denylist:
                print(f"lat {lat} in denylist, skipping..")
                continue
        # print(lat,lon)
        lats.append(float(lat))
        lons.append(float(lon))

    print(f"input len {len(data[1:])} filtered len {len(lats)}")

    with open("city_elev_stdev_exact.csv",'r', encoding="ISO-8859-1") as city_data_file:
        city_data_file.readline() # skip header
        for row in city_data_file:

            # a quick and dirty way to get the row that matches city_id, assuming city is unique for a country code
            if int(geonameid) == int(row.split(",")[0]):
                lat = float(row.split(",")[3])
                lon = float(row.split(",")[4])
                print(row,lat,lon)
                city_name = row.split(",")[1]
                break
    BL,TR = get_square_corners(lat, lon, side_length=8000)
    elevation_data, _, _, _ = read_exact_elevation_from_zip_file(BL[0],BL[1],TR[0],TR[1])
    print(f"elevation data {elevation_data} ||")
    m = len(elevation_data)
    n = len(elevation_data[0])

    x_vals, y_vals = gps_to_array_coords(lats, lons, BL, TR, m, n)



    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_data, cmap='gray', vmin=np.min(elevation_data), vmax=np.max(elevation_data))
    cbar = plt.colorbar()
    cbar.set_label('Elevation (m)',fontsize = 20)
    cbar.ax.tick_params(labelsize=20)
    # plt.title(f'Elevation Map for {city_name}')
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    plt.tight_layout()
    plt.scatter(x_vals,y_vals, facecolors='white', edgecolors='red', s=100)

    filename = f"nodes_and_elevation_{geonameid}_{city_name}_{"denylist" if denylist_filter else "normal"}.png"
    plt.savefig(filename, dpi=300)

    plt.show()

    # test_and_plot_selection()

def plot_timespan_and_error():
    #SEA30 dataset
    output = [{'timespan-[1635896268, 1638524556]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2604.4692'],
                                                     [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1030.0905'],
                                                     [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test',
                                                      '1255.3203'],
                                                     [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '1667.6445'],
                                                     [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2368.3657'],
                                                     [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                      '0.2testsizegrid2_test', '2017.2415'],
                                                     [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2374.4421'],
                                                     [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1056.4209'],
                                                     [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                      '0.2testsizegrid5_test', '1318.2003']]}, {
                  'timespan-[1638525049, 1641153337]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2568.7869'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '934.3393'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1162.8083'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2500.169'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1177.0708'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1465.612']]}, {
                  'timespan-[1641154285, 1643782573]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '1984.0078'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1211.2169'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1284.555'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2823.754'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '3092.4683'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '2911.5425'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2417.308'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1259.3456'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1634.1616']]}, {
                  'timespan-[1643782688, 1646410976]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2243.7197'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1245.4286'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1549.7991'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2499.3372'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '3171.8716'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '3381.7092'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2704.6548'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1935.1259'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '2207.2634']]}, {
                  'timespan-[1646412372, 1649040660]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2487.181'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1143.9791'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1616.763'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2717.722'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '2885.149'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                       '0.2testsizegrid2_test', '3370.9558'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2463.5867'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1697.067'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                       '0.2testsizegrid5_test', '2106.498']]}, {
                  'timespan-[1649041659, 1651669947]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2231.6682'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1377.3357'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1528.1565'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2083.866'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '2671.008'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                       '0.2testsizegrid2_test', '2149.6785'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2730.6206'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1212.6995'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1672.2434']]}, {
                  'timespan-[1651670093, 1654298381]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2474.6655'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1069.3596'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1122.98'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2409.5686'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '2994.0017'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '3207.68'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2466.938'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1143.7452'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1500.6298']]}, {
                  'timespan-[1654299421, 1656927709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2429.9832'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1242.863'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1511.3627'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2281.4714'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '3008.55'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                      '0.2testsizegrid2_test', '3139.0764'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2495.0088'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1102.6265'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1293.7515']]}, {
                  'timespan-[1656928416, 1659556704]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2315.2664'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1155.8884'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1330.5139'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '3387.3765'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '2774.0842'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '3233.8882'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2730.0095'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1275.272'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                       '0.2testsizegrid5_test', '1376.2615']]}, {
                  'timespan-[1659558771, 1662187059]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2256.7173'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1379.244'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1455.4543'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2381.4404'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '1720.401'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                       '0.2testsizegrid2_test', '2162.1206'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2645.0146'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1406.4465'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1864.7751']]}, {
                  'timespan-[1662188017, 1664816305]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2139.0884'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1005.4674'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1146.7332'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2893.9512'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '3490.4988'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '3484.2815'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2710.7673'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1115.5552'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1359.1348']]}, {
                  'timespan-[1664817247, 1667445535]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2388.3408'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1206.6405'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1449.9829'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2807.5122'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1290.8182'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1553.3966']]}, {
                  'timespan-[1667445931, 1670074219]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2423.734'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1335.5262'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1272.499'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2209.2222'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '1433.5565'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '1729.4646'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2515.7202'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1480.6077'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1568.2676']]}, {
                  'timespan-[1670074662, 1672702950]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2527.7263'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1088.0005'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1162.1332'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2709.9949'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '3497.4722'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '4152.147'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2418.784'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1437.2137'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1923.0874']]}, {
                  'timespan-[1672703323, 1675331611]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2690.1243'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '734.4363'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1028.7834'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '2087.3506'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '2325.1606'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid2_test', '1942.6198'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2657.0215'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1258.7013'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1429.3624']]}, {
                  'timespan-[1675332204, 1677960492]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2529.4575'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '928.48987'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1032.0994'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2426.2388'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1245.6614'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1805.3197']]}, {
                  'timespan-[1677960546, 1680588834]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2367.5747'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1119.4985'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '1262.8776'],
                                                        [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test',
                                                         '4155.9883'],
                                                        [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test',
                                                         '4249.502'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()',
                                                                       '0.2testsizegrid2_test', '4011.8782'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '3082.653'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '958.059'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                      '0.2testsizegrid5_test', '2577.5186']]}, {
                  'timespan-[1680588975, 1683217263]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2725.775'],
                                                        [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1415.4572'],
                                                        [9, 'random', 0, 'SlicedEarthMoversDistance()',
                                                         '0.2testsize_test', '2036.613'],
                                                        [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test',
                                                         '2830.209'],
                                                        [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test',
                                                         '1696.9727'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()',
                                                                        '0.2testsizegrid5_test', '1678.135']]}]
    # San Francisco
    output = [{'timespan-[1635890982, 1638519270, 13]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7664.6943'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2074.319'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2102.7546'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4337.6606'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2480.7537'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2648.4497'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7936.956'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1932.8818'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1978.0084']]}, {'timespan-[1638519276, 1641147564, 51203]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7510.135'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2185.816'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2269.6978'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4689.3965'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3296.69'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3610.1096'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8123.4375'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2240.2036'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2378.7742']]}, {'timespan-[1641147650, 1643775938, 60709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6836.775'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2274.4893'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2260.7974'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4367.2065'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3009.8367'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3308.807'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7726.193'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2593.6992'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2633.5803']]}, {'timespan-[1641147650, 1643775938, 60709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7586.0024'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2268.738'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2251.4219'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4226.4473'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2997.107'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3388.1987'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7497.978'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2696.6821'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2674.945']]}, {'timespan-[1641147650, 1643775938, 60709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7701.6304'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2273.674'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2145.022'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4399.6343'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2981.9714'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3126.792'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7519.238'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2609.1133'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2703.4675']]}, {'timespan-[1643775940, 1646404228, 62572]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7590.315'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2046.9113'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2095.3562'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4414.7944'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3075.413'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3218.448'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7316.951'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2319.7122'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2352.1177']]}, {'timespan-[1643775940, 1646404228, 62572]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7902.884'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2042.1448'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2133.5574'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4802.3506'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3075.9026'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3191.5554'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7985.3174'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2300.5583'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2428.6208']]}, {'timespan-[1646404229, 1649032517, 56950]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '8123.7026'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2311.8357'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2325.9226'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4770.3354'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2709.4475'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2982.136'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7741.288'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2315.5415'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2359.2976']]}, {'timespan-[1649032732, 1651661020, 42181]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7543.5454'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2083.7468'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2157.9172'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4522.353'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2510.0154'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2830.5378'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7676.3496'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2085.0437'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2179.0542']]}, {'timespan-[1651661105, 1654289393, 23423]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7502.5747'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2194.295'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2264.981'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4901.5376'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2843.1663'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3570.207'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7964.311'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2166.8943'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2259.4685']]}, {'timespan-[1654289759, 1656918047, 17242]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7281.8076'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2131.9172'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2292.802'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4423.437'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2531.468'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2981.6494'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8191.4155'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1975.7416'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2083.9846']]}, {'timespan-[1654289759, 1656918047, 17242]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7493.5205'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2131.4387'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2195.0251'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '5231.3906'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2531.0757'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2646.099'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8113.6396'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1974.5251'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2072.7708']]}, {'timespan-[1654289759, 1656918047, 17242]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7524.614'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2129.6704'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2209.207'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '5243.4087'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2535.504'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2743.1892'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7624.3623'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1972.935'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1991.05']]}, {'timespan-[1656918908, 1659547196, 16679]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6836.802'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2310.7268'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2206.5735'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4560.984'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2709.3955'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2779.7427'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7960.789'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2071.4863'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2131.2466']]}, {'timespan-[1656918908, 1659547196, 16679]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6894.234'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2121.2642'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2152.0032'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4849.08'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2685.8787'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2995.5515'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8194.018'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2080.0112'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2097.183']]}, {'timespan-[1656918908, 1659547196, 16679]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7088.8525'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2126.06'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2161.0334'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4748.9897'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2686.0845'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2879.966'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7210.3335'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2094.595'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2186.8943']]}, {'timespan-[1659547633, 1662175921, 12623]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7522.202'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2125.8774'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2229.5493'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4235.9956'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2824.72'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3197.524'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7633.822'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2353.2373'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2365.2893']]}, {'timespan-[1659547633, 1662175921, 12623]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7650.0737'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2105.7917'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2231.508'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4775.5146'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2838.4736'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2913.516'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7742.76'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2355.129'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2366.476']]}, {'timespan-[1659547633, 1662175921, 12623]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7679.4146'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2104.5415'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2187.527'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4775.0137'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2825.8337'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2984.312'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7877.6685'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2349.8484'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2395.558']]}]

    # San Diego
    # output = [{'timespan-[1501920252, 1504548540, 1]': []}, {'timespan-[1635896053, 1638524341, 2]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3148.3477'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2645.2021'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3391.5127'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4498.968'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3404.7593'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5191.675'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3845.1052'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3171.9802'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '4572.058']]}, {'timespan-[1638524621, 1641152909, 3004]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3061.8303'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2315.369'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2890.3982'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4685.2515'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3516.8513'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5450.52'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3237.503'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3191.6804'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2947.3167']]}, {'timespan-[1641154548, 1643782836, 3929]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2982.41'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2871.9197'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2932.4424'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '3968.1948'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2863.5564'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4934.7915'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3187.2993'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2191.6687'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2830.21']]}, {'timespan-[1643783404, 1646411692, 4324]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3198.0015'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3129.7915'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3574.4487'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4634.913'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3012.5662'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5148.019'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3123.6848'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2818.1997'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3759.8516']]}, {'timespan-[1646412680, 1649040968, 3458]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2900.6377'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2939.4922'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3018.8022'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4217.2114'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3081.5376'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4663.212'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3421.3594'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2959.1687'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3749.4995']]}, {'timespan-[1649042174, 1651670462, 2514]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2953.9595'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2633.343'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3189.4375'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4409.0503'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2853.9453'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5065.988'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3472.2244'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2953.314'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3562.8762']]}, {'timespan-[1651671228, 1654299516, 1482]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3109.8518'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2512.5437'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3206.727'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4692.47'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3553.031'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5358.694'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3371.0317'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2554.4268'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '4498.555']]}, {'timespan-[1654301298, 1656929586, 1096]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3201.6216'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2586.846'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2659.0054'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4342.9473'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2593.4363'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3553.7517'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3095.4556'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2871.2053'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3140.569']]}, {'timespan-[1656946921, 1659575209, 1075]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3096.2495'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3303.1782'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3720.4595'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4783.321'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2958.2476'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5589.598'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3655.8472'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2832.7935'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '4512.353']]}, {'timespan-[1659577116, 1662205404, 919]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3140.5862'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2955.2012'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '4181.419'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4306.379'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3880.524'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5718.4287'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3695.457'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3209.2393'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3565.361']]}, {'timespan-[1662212994, 1664841282, 1220]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3295.696'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2795.5117'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3407.1086'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4516.8955'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3018.7852'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '6075.65'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3426.1914'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2659.6365'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3976.929']]}, {'timespan-[1664841650, 1667469938, 1280]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3114.5906'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2979.2402'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3443.19'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4634.2695'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3617.5742'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5123.612'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3632.2688'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2471.0847'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3717.9956']]}, {'timespan-[1667477540, 1670105828, 1034]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3166.6367'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2647.123'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3204.629'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4291.315'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3171.0193'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5166.5737'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3313.906'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2532.8425'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3297.0261']]}, {'timespan-[1670112648, 1672740936, 1284]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3210.0'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3340.5981'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2997.1921'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4023.0806'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3478.9565'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4652.4077'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3397.358'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2699.5806'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3173.7993']]}, {'timespan-[1672747521, 1675375809, 1459]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2809.819'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2073.4316'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2712.1982'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4058.6165'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3393.6099'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4470.3354'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3689.6902'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2830.101'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3628.509']]}, {'timespan-[1675375955, 1678004243, 1239]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3036.6934'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2673.0625'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2667.8801'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4672.7476'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3052.6108'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5414.1567'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3723.6094'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3293.3928'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3433.947']]}, {'timespan-[1678004317, 1680632605, 4608]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2898.807'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2531.3728'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3071.726'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '3782.2676'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3297.6624'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4647.9526'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3816.8904'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3187.3027'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3235.964']]}, {'timespan-[1680632608, 1683260896, 4559]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3464.3494'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3162.2454'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3475.143'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4191.8374'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '6730.183'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5026.0093'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3732.6533'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3106.4985'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3819.3877']]}]

    split_types = ['random', 'grid2', 'grid5']
    model_types = ['CoMLoss()', 'MSELoss()', 'SlicedEarthMoversDistance()']

    results_dict = {}


    # for min error
    min_y_vals = []
    for span in output:  # for timespan results
        # print(f"span {span}")
        # print(f"span.keys() = {span.keys()}")
        min_error = 9999.0
        span_key = list(span.keys())[0]
        # print(f"span[span_key] {span[span_key]}")
        for result in span[span_key]:
            # print(result)
            if float(result[-1]) < min_error:
                min_error = float(result[-1])
                min_key = span_key
        print(f"{span_key} : {min_error}")
        min_y_vals.append(min_error)

    # for error in all models
    for span_dict in output:
        for span,list_of_records in span_dict.items():
            results_dict[span] = {split_type: {model_type: None for model_type in model_types} for split_type in split_types}
            for record in list_of_records:
                # print(span,record)
                _, split_type, _, model_type, _, error_value = record
                if split_type in split_types and model_type in model_types:
                    results_dict[span][split_type][model_type] = float(error_value)

    results_dict2= {}
    tmp_results_dict = {}
    for split_type in split_types:
        for model_type in model_types:
            print(model_type, split_type)
            results_dict2[(model_type + '_' + split_type)] = []
            tmp_results_dict[(model_type + '_' + split_type)] = None

    print(results_dict2)

    for span_dict in output:
        for span,list_of_records in span_dict.items():
            count = 0
            tmp_results_dict = {key: None for key in tmp_results_dict}
            for record in list_of_records:
                _, split_type, _, model_type, _, error_value = record
                tmp_results_dict[model_type + '_' + split_type] = error_value
                # print(model_type + '_' + split_type)
                # print(tmp_results_dict[model_type + '_' + split_type])
                count += 1
            print(f"{count} records in {span}")
            for result in tmp_results_dict.keys():
                # print(result)
                if tmp_results_dict[result]:
                    results_dict2[result].append(float(tmp_results_dict[result]))
                else:
                    results_dict2[result].append(None)
                    print(f"appended None to {results_dict2[result]}")

    plot_colors = {}
    color_list = ['darkblue','steelblue','slateblue']
    color_index = -1
    for model_type in model_types:
        color_index += 1
        for split_type in split_types:
            plot_colors[model_type + '_' + split_type] = color_list[color_index]



    # for key,y_vals in results_dict2.items():
    #     print(key,plot_colors[key])
    #     x_vals = range(len(y_vals))
    #     # try:
    #     #     slope, intercept = np.polyfit(x_vals, y_vals, 1)
    #     #     regression_line = slope * x_vals + intercept
    #     #     plt.plot(x_vals, regression_line, color=plot_colors[key], linestyle='--')
    #     # except:
    #     #     pass
    #     plt.scatter(x_vals, y_vals, color=plot_colors[key], alpha=0.8, edgecolors='black')
    x_vals = range(len(min_y_vals))
    plt.figure(figsize=(6, 2), dpi=300)
    plt.scatter(x_vals,min_y_vals, color='blue', alpha=1.0, edgecolors='black')
    # min regression line
    try:
            slope, intercept = np.polyfit(x_vals, min_y_vals, 1)
            regression_line = slope * x_vals + intercept
            plt.plot(x_vals, regression_line, color='blue', linestyle='dotted')
            print(f"slope : {slope}")
    except:
        pass

    # Add titles and labels
    plt.xticks([])
    plt.title(f'Change in Error Over Time in San Francisco')
    plt.ylabel('Mean Error δ (m)')
    plt.xlabel('Time')


    filename = f"Error_over_time_SF30.png"
    plt.savefig(filename, dpi=300)
    plt.show()

def temporal_spread_all():
    # file_path = r'C:\Users\ps\Helium_Data\all_nz_distances_di9.csv'
    file_path = r'C:\Users\ps\Helium_Data\all_nz_v2_di9.csv'

    timestamps = []
    count_dict = {}
    # Read the file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i % 10000000 == 0 and i > 0:
                print(f"Processed {i} lines...")
            if row['timestamp'] not in count_dict.keys():
                count_dict[row['timestamp']] = 1
            else:
                count_dict[row['timestamp']] += 1
            timestamps.append(int(row['timestamp']))


    # Get the smallest and largest timestamps
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)

    actual_min_timestamp = 0 #1564436733

    # Convert timestamps to datetime objects and extract year and month
    # dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    dates = []
    months = []
    total_count = 0
    bad_count = 0
    for ts in timestamps:
        total_count += 1
        if ts > actual_min_timestamp:
            try:
                date = datetime.fromtimestamp(ts)
                month = (date.year, date.month)
            except:
                print(f"Failed conversion to dates for {ts}")
                continue
            try:
                months.append(month)
            except:
                print(f"Could not append {month} to month list")
        else:
            bad_count += 1


    # months = [(date.year, date.month) for date in dates]



    print(dates[:10],dates[-10:])
    print(months[:10],months[-10:])

    # Count entries per month
    monthly_counts = Counter(months)

    sorted_months = sorted(monthly_counts.keys())
    counts = [monthly_counts[month] for month in sorted_months]
    print(counts)

    # Convert (year, month) tuples to readable labels
    labels = [f"{year}-{month:02d}" for year, month in sorted_months]

    # Plot the data
    plt.figure(figsize=(6, 2.5), dpi=300)
    plt.xticks([])
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Months')
    plt.ylabel('Transmit Event Count')
    plt.title('Temporal Spread of All Helium PoC Data')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = "Temporal_Spread_all_data.png"
    plt.savefig(filename,dpi=300)
    plt.show()


    code.interact(local=locals())

if __name__ == '__main__':
    # temporal_spread_all()
    # plot_timespan_and_error()
    # plot_denylist_and_error()
    # save_elevationstdev_vs_error()
    # plot_and_save_samples_vs_error()
    # plot_elevationstdev_vs_error('20240925_normal_error_vs_elev_stdev_exact.csv')
    # genameid_list = [4791259,5520993,625144,618426]
    # best worst US cities genameid
    # genameid_list = [4174757, 4691930, 4791259, 4951305, 5139568, 5150529, 5308655, 5350937, 5454711, 5520993]
    # best worse EUro cities genameid
    genameid_list = [3067696, 3165524, 3172394, 3191281, 611717, 616052, 618426, 625144, 709717, 709930]

    for geonameid in genameid_list:
        plot_node_locations_on_elevation(geonameid,denylist_filter = False)