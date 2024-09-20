
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import re
import csv
import numpy as np
from city_elevation_processor import test_and_plot_selection, get_square_corners, read_exact_elevation_from_zip_file


def plot_and_save_samples_vs_error(merged_file = '20240912_deny_list_merged_output.csv' ):
    # Read data from CSV file
    #data = pd.read_csv('deny_list_merged_output.csv')
    data = pd.read_csv(merged_file)

    # Extracting the first column as y values and the second column as x values
    y = data.iloc[:, 3] # samples_count
    x = data.iloc[:, 2] # error

    plt.scatter(x, y, c='blue', alpha=0.2, edgecolors='black')

    # Add titles and labels
    plt.title(f'Mean Error vs Sample Quantity in 5000+ Cities')
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

# Fairland,US,39.07622,-76.95775,44.56766372846624 vs 2024_09_11-23_02_46-results.txt,Adeje__ES8.csv,1434.1417,4511
# can take city_elev_stdev.csv or city_elev_stdev_exact.csv with more precise selection area
def save_elevationstdev_vs_error(elev_stdev_file = 'city_elev_stdev_exact.csv',
                                          error_file = '20240912_deny_list_merged_output.csv' ):
    elev_stdev_data = pd.read_csv(elev_stdev_file, encoding="ISO-8859-1")
    error_data = pd.read_csv(error_file, encoding="ISO-8859-1")

    outfile = '20240912_error_vs_elev_stdev_exact.csv'


    with open(outfile, 'w',newline='') as csvfile:
        print(f"opened {outfile} for writing")
        writer = csv.writer(csvfile)
        header = ["city_id","error","stdev"]
        writer.writerow(header)
        for index,row in error_data.iterrows():
            # print(f"index {index} row{row}")
            try:
                city_id = re.split(r'\d+', row['city_id'])[0]
                # print(city_id)
            except Exception as e:
                print(f"Failed to retrieve city_id for {row['city_id']} with error {e}")
                continue
            #print(city_id)
            for index2, row2 in elev_stdev_data.iterrows():
                try:
                    # print(f"index2 {index2} row2{row2}")
                    # print(row2['city'])
                    country = row2['country']
                    if not isinstance(country,str):
                        country = "NA"
                    try:
                        city_id2 = row2['city'].replace(' ','_') + '__' + country
                        # print(city_id2)
                    except Exception as e:
                        print(f"Failed to retrieve city_id2 for {row2['city']} and {country} with error {e}")
                        continue # ignore the city if the country code is absent
                    if city_id == city_id2:
                        # print(city_id, city_id2)
                        # print(f"{city_id},{row['error']},{row2['stdev_elev']}")
                        try:
                            row_towrite = [city_id,row['error'],row2['stdev_elev']]
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
    plt.title(f'Mean Error vs Elevation σ in 5000+ Cities')
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
        y = int((lat - lat_min)/ lat_range * (m-1))
        x = int((lon - lon_min) / lon_range * (n - 1))
        x_vals.append(x)
        y_vals.append(y)
    if count > 0:
        print(f"WARNING: {count}/{len(latitudes)} points fell outside of lat/lon bounds")
    return x_vals, y_vals

def plot_node_locations_on_elevation(city_id = "Dallas__US8",
    data_directory = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240912_15000cities_denylist\generated"):
    filepath = f"{data_directory}\\{city_id}.csv"
    lats = []
    lons = []
    with open(filepath,'r') as input_data_file:
        data = input_data_file.readlines()
    for line in data[1:]: # skip header line
        print(line)
        lats.append(float(line.split(',')[1]))
        lons.append(float(line.split(',')[2]))


    with open("city_elev_stdev_exact.csv",'r', encoding="ISO-8859-1") as city_data_file:
        for row in city_data_file:
            # a quick and dirty way to get the row that matches city_id, assuming city is unique for a country code
            if city_id.split("__")[0].replace("_"," ") == row.split(",")[1]:
                lat = float(row.split(",")[3])
                lon = float(row.split(",")[4])
                print(row,lat,lon)
                break
    BL,TR = get_square_corners(lat, lon, side_length=8000)
    elevation_data, _, _, _ = read_exact_elevation_from_zip_file(BL[0],BL[1],TR[0],TR[1])
    print(elevation_data)
    m = len(elevation_data)
    n = len(elevation_data[0])

    x_vals, y_vals = gps_to_array_coords(lats, lons, BL, TR, m, n)

    # x_vals = [(x-float(BL[1])) for x in lats]
    # y_vals = [(y-float(BL[0])) for y in lons]

    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_data, cmap='terrain', vmin=np.min(elevation_data), vmax=np.max(elevation_data))
    plt.colorbar(label='Elevation (m)')
    plt.title(f'Elevation Map for {city_id}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.scatter(x_vals,y_vals, c='blue', alpha=0.2, edgecolors='black')
    plt.show()

    # test_and_plot_selection()


if __name__ == '__main__':
    # plot_and_save_samples_vs_error()
    #save_elevationstdev_vs_error()
    #plot_elevationstdev_vs_error()
    plot_node_locations_on_elevation() # TODO this is maching up the wrong Dallas's.. need to fix this and make unique
    # TODO will adjust this with the new cities format that has geonameid in the filename