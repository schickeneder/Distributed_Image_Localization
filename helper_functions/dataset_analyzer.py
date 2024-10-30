# This contains functions used to analyze a dataset that may be used as inputs to a model
import numpy as np
import glob
import pandas as pd
import cupy as cp
import code

def haversine_distance2(lat1, lon1, lat2, lon2, chunk_size=100):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface using the Haversine formula.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = (
        cp.radians(cp.array(lat1, dtype=cp.float32)),
        cp.radians(cp.array(lon1, dtype=cp.float32)),
        cp.radians(cp.array(lat2, dtype=cp.float32)),
        cp.radians(cp.array(lon2, dtype=cp.float32))
    )
    # Radius of the Earth in kilometers
    R = 6371.0

    # Initialize an empty list to store distance results from each chunk
    distances = []

    # Process distances in chunks to manage GPU memory usage
    for i in range(0, len(lat1), chunk_size):
        # Get the current chunk
        lat1_chunk = lat1[i:i + chunk_size]
        lon1_chunk = lon1[i:i + chunk_size]
        lat2_chunk = lat2[i:i + chunk_size]
        lon2_chunk = lon2[i:i + chunk_size]

        # Compute haversine formula components
        dlon = lon2_chunk - lon1_chunk
        dlat = lat2_chunk - lat1_chunk
        a = cp.sin(dlat / 2) ** 2 + cp.cos(lat1_chunk) * cp.cos(lat2_chunk) * cp.sin(dlon / 2) ** 2
        c = 2 * cp.arcsin(cp.sqrt(a))

        # Calculate distance and transfer to CPU memory
        distance_chunk = (c * R).get()  # Transfer to CPU to free GPU memory
        distances.append(distance_chunk)

        # Free GPU memory for this chunk
        del lat1_chunk, lon1_chunk, lat2_chunk, lon2_chunk, dlon, dlat, a, c
        cp.cuda.Stream.null.synchronize()  # Ensures memory is freed

    # Concatenate all chunks into a single array on the CPU
    final_distances = cp.concatenate([cp.array(chunk) for chunk in distances]).get()
    return final_distances

def compute_distances2(coords_list1, coords_list2,threshold=10000):
    """
    Compute distances between every coordinate in coords_list1 to every coordinate in coords_list2.

    Parameters:
    - coords_list1 (tuple or list): Tuple or list of two CuPy arrays containing latitude and longitude coordinates.
    - coords_list2 (tuple or list): Tuple or list of two CuPy arrays containing latitude and longitude coordinates.

    Returns:
    - distances (cupy.ndarray): 2D array containing distances between each pair of coordinates.
    """
    # code.interact(local=locals())

    # print(coords_list1)
    # print(coords_list2)
    latitudes1 = coords_list1[0]
    longitudes1 = coords_list1[1]
    latitudes2 = coords_list2[0]
    longitudes2 = coords_list2[1]

    # latitudes1, longitudes1 = coords_list1
    # latitudes2, longitudes2 = coords_list2

    # Expand dimensions to perform broadcasting
    latitudes1 = latitudes1[:, cp.newaxis]
    longitudes1 = longitudes1[:, cp.newaxis]
    latitudes2 = latitudes2[cp.newaxis, :]
    longitudes2 = longitudes2[cp.newaxis, :]

    # Compute distances using Haversine formula
    distances = haversine_distance(latitudes1, longitudes1, latitudes2, longitudes2)

    #print(f"distances {len(distances)} with shape {distances.shape}")

    filtered_distances = distances[distances < threshold]

    #print(f"filtered_distances {len(filtered_distances)} with shape {filtered_distances.shape}")

    #return sum of filtered_distances
    return cp.sum(filtered_distances), len(filtered_distances), cp.mean(filtered_distances), cp.median(filtered_distances)



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface using the Haversine formula.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = cp.radians(lat1), cp.radians(lon1), cp.radians(lat2), cp.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = cp.sin(dlat / 2) ** 2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon / 2) ** 2
    c = 2 * cp.arcsin(cp.sqrt(a))
    r = 6371000  # Radius of the Earth in meters
    return c * r

def compute_distances(coords_list1, coords_list2,threshold=10000):
    """
    Compute distances between every coordinate in coords_list1 to every coordinate in coords_list2.

    Parameters:
    - coords_list1 (tuple or list): Tuple or list of two CuPy arrays containing latitude and longitude coordinates.
    - coords_list2 (tuple or list): Tuple or list of two CuPy arrays containing latitude and longitude coordinates.

    Returns:
    - distances (cupy.ndarray): 2D array containing distances between each pair of coordinates.
    """

    latitudes1, longitudes1 = coords_list1
    latitudes2, longitudes2 = coords_list2

    # Expand dimensions to perform broadcasting
    latitudes1 = latitudes1[:, cp.newaxis]
    longitudes1 = longitudes1[:, cp.newaxis]
    latitudes2 = latitudes2[cp.newaxis, :]
    longitudes2 = longitudes2[cp.newaxis, :]

    # Compute distances using Haversine formula
    distances = haversine_distance(latitudes1, longitudes1, latitudes2, longitudes2)

    # print(f"distances {len(distances)} with shape {distances.shape}")

    # filtered_distances = distances[distances < threshold]

    # print(f"filtered_distances {len(filtered_distances)} with shape {filtered_distances.shape}")

    #return filtered_distances

    return distances

def compute_distances_split(coordinates_list, span):
    all_distances = []
    for i in np.arange(0, len(coordinates_list)-1, span):
        print(f"splitting distances between {i} and {i+span}")
        all_distances.append(compute_distances(coordinates_list[i:i+span], coordinates_list[i:i+span]))
    print(np.array(all_distances).mean())

def compute_distances_between_locations(coordinates_list):
    coords = cp.radians(cp.array(coordinates_list))  # Convert degrees to radians

    # Separate latitude and longitude
    latitudes = coords[:, 0]
    longitudes = coords[:, 1]

    # Calculate the pairwise distance matrix using the Haversine formula
    # Earth's radius in kilometers
    R = 6371.0

    # Calculate differences between each pair
    lat_diffs = latitudes[:, cp.newaxis] - latitudes
    lon_diffs = longitudes[:, cp.newaxis] - longitudes

    # Haversine formula
    a = cp.sin(lat_diffs / 2) ** 2 + cp.cos(latitudes)[:, cp.newaxis] * cp.cos(latitudes) * cp.sin(lon_diffs / 2) ** 2
    distances = 2 * R * cp.arcsin(cp.sqrt(a))

    distances_cpu = distances.get()
    # Convert the distance matrix to a list (optional, for readability)
    print(distances_cpu.mean())
    distances_list = distances_cpu.tolist()
    return distances_list

def get_tx_density_from_generated(directory_path):
# computes data density for all generated dataset files within a directory like:
# "4281730_Wichita__US8.csv"
# time	lat1	lon1	lat2	lon2	txpwr	rxpwr
# 1675353231	37.67658103	-97.33914135	37.68358968	-97.33383319	27	-114
    csv_files = glob.glob(f"{directory_path}/*.csv")
    for file in csv_files:
        print(file)
        data = pd.read_csv(file)
        coordinates = data[['lat1','lon1','lat2','lon2']].drop_duplicates().values.tolist()

        # compute_distances_between_locations(coordinates)
        tx_lats = cp.array([lat1 for lat1, lon1, lat2, lon2 in coordinates])
        tx_lons = cp.array([lon1 for lat1, lon1, lat2, lon2 in coordinates])
        rx_lats = cp.array([lat2 for lat1, lon1, lat2, lon2 in coordinates])
        rx_lons = cp.array([lon2 for lat1, lon1, lat2, lon2 in coordinates])

        try:
            dist_sum, dist_length, dist_mean, dist_med = compute_distances2([tx_lats, tx_lons],[tx_lats,tx_lons],1000)
        except:
            print(f"Failed for file {file}")
            continue
        print(f"dist_sum {dist_sum} dist_length {dist_length} dist_mean {dist_mean} dist_median {dist_med}")
    code.interact(local=locals())


if __name__ == '__main__':
    get_tx_density_from_generated(r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240926_top60us_normal2\generated")