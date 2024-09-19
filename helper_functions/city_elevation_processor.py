import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import zipfile
import csv
import threading
import io
import re
import pandas as pd
import math

SRTM_DICT = {'SRTM1': 3601, 'SRTM3': 1201}

# Get the type of SRTM files or use SRTM3 by default
SRTM_TYPE = os.getenv('SRTM_TYPE', 'SRTM1')
SAMPLES = SRTM_DICT[SRTM_TYPE]

SRTM_DIR = r"C:\Users\ps\Helium_Data\SRTM Elevation Data\SRTMGL1_003-20240911_232824"

city_list = r"cities15000extracted_citycountry_lat_lon.txt"

RAW_CITIES = r"cities15000_raw.txt"


# read raw city list from ....?
# create list of dictionaries for each city
def read_city_list(file_path=RAW_CITIES):
    cities = []
    with open(file_path, mode='r', encoding="ISO-8859-1") as file:
        reader = csv.reader(file, delimiter='\t')  # Use tab delimiter
        next(reader)  # Skip header if present
        for row in reader:
            if len(row) >= 15:  # Ensure there are enough columns
                city = {
                    'geonameid': row[0],
                    'name': row[1],
                    'alternate_names': row[2],
                    'latitude': row[4],
                    'longitude': row[5],
                    'country': row[8],
                    'population': row[-5],
                    'timezone': row[-2],
                }
                cities.append(city)
    return cities


def print_city_info(cities):
    for city in cities:
        try:
            print(f'("{city['name']}, {city['country']}",{city['latitude']},{city['longitude']}),')
        except:
            continue
        # print(f"City: {city['name']}, Country: {city['country']}, "
        #       f"Latitude: {city['latitude']}, Longitude: {city['longitude']}, "
        #       f"Population: {city['population']}, "
        #       f"Timezone: {city['timezone']}")


# takes cities as list of dicts, extracts lat/long then finds resulting elev stdev
# returns list containing rows of [cityname,country,lat,lon,elev_std]
def generate_elev_stdev_list(cities):
    new_list = []
    total = len(cities)
    count = 0
    for city in cities:
        if count % 1000 == 0:
            print(f"{count}/{total} rows processed")
        cityname = city['name']
        country = city['country']
        lat = float(city['latitude'])
        lon = float(city['longitude'])
        results = read_elevation_from_zip_file(lat, lon)
        if results:
            stdev = results[1]
            new_list.append([cityname, country, lat, lon, stdev])
        count += 1
    return new_list

# get BL and TR coordinates for a square area centered at lat, lon with length side_length (meters)
def get_square_corners(lat, lon, side_length=8000):
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

def process_city(city, results, index):
    geonameid = city['geonameid']
    cityname = city['name']
    country = city['country']
    lat = float(city['latitude'])
    lon = float(city['longitude'])
#    result = read_elevation_from_zip_file(lat, lon)
    bl,tr = get_square_corners(lat, lon)
    result = read_exact_elevation_from_zip_file(bl[0],bl[1],tr[0],tr[1])
    if result:
        stdev_elev = result[1]
        min_elev = result[2]
        max_elev = result[3]
        results[index] = [geonameid,cityname, country, lat, lon, stdev_elev, min_elev, max_elev]


def generate_elev_stdev_list2(cities):
    total = len(cities)
    results = [None] * total
    threads = []

    for index, city in enumerate(cities):
        if index % 1000 == 0:
            print(f"{index}/{total} rows processed")
        thread = threading.Thread(target=process_city, args=(city, results, index))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Filter out None values in case some cities didn't return results
    return [result for result in results if result is not None]


# returns filename for an SRTM.hgt tile, but leaves off extension '.hgt'
def get_hgt_filename(lat, lon):
    if lat > 0:
        filenamepart1 = 'N' + f"{abs(int(lat)):02d}"
    else:
        filenamepart1 = 'S' + f"{abs(int(lat)-1):02d}" # filename is always SW corner
    if lon > 0:
        filenamepart2 = 'E' + f"{abs(int(lon)):03d}"
    else:
        filenamepart2 = 'W' + f"{abs(int(lon)-1):03d}"  # filename is always SW corner

    target_filename = filenamepart1 + filenamepart2

    return target_filename


# returns all elevation data for tile containing lat/lon
def read_elevation_from_zip_file(lat, lon, path=SRTM_DIR):
    target_filename = get_hgt_filename(lat, lon)

    for filename in os.listdir(path):
        if target_filename in filename and filename.endswith(".zip"):
            zip_path = os.path.join(SRTM_DIR, filename)
            break

    target_filename += ".hgt"
    #print(target_filename)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(target_filename) as hgt_data:
                try:
                    # Read the data into a BytesIO object
                    byte_stream = io.BytesIO(hgt_data.read())

                    # Read binary data directly from the BytesIO stream
                    byte_stream.seek(0)  # Make sure to rewind the BytesIO object to the beginning
                    elevations = np.frombuffer(
                        byte_stream.getvalue(),  # Get the binary data
                        dtype=np.dtype('>i2'),  # Data type
                        count=SAMPLES * SAMPLES  # Length
                    )

                except Exception as e:
                    print(
                        f"Couldn't process {target_filename} and elevations {elevations} of len {len(elevations)} because {e}")
        return elevations.reshape((SAMPLES, SAMPLES)), np.std(elevations), np.min(elevations), np.max(elevations)

    except Exception as e:
        print(f"Couldn't open zip file {target_filename} because {e}")
        return None


# returns all elevation data for tiles bounded by (lat,lon) and (lat2,lon2)
# where (lat,lon) is SW coordinate and (lat2,lon2) is NE coordinate
# if (lat,lon) is near the N and/or E edge of tile, then may need to load up to 3 additional tiles to capture the area
# we are assuming that the area will not span more than 4 tiles.. we can add a check for this somewhere
# num_tiles is how many on a side we want to consider, default 2, TODO: test with more than 2
def read_exact_elevation_from_zip_file(lat, lon, lat2, lon2, path=SRTM_DIR, num_tiles=2):
    try:
        if int(lat) == int(lat2) and int(lon) == int(lon2):  # case 1 (default - it's bounded in 1 tile)
            tile, _, _, _ = read_elevation_from_zip_file(lat, lon, path=path)

            # start_row = SAMPLES + 1 - int(abs(lat2 - int(lat)) * (SAMPLES - 1))
            # end_row = SAMPLES + 1 - int(abs((lat - int(lat)) * (SAMPLES - 1)))
            # start_col = SAMPLES - int(abs((lon - int(lon)) * (SAMPLES - 1)))
            # end_col = SAMPLES - int(abs((lon2 - int(lon)) * (SAMPLES - 1)))

            start_row = SAMPLES + 1 - int((lat2 - np.floor(lat)) * (SAMPLES - 1))
            end_row = SAMPLES + 1 - int((lat - np.floor(lat)) * (SAMPLES - 1))
            start_col = int((lon - np.floor(lon)) * (SAMPLES - 1))
            end_col = int((lon2 - np.floor(lon)) * (SAMPLES - 1))

            #print(start_row, end_row, start_col, end_col)
            #print(len(tile), len(tile[0]))

            selection = tile[start_row:end_row, start_col:end_col]

            #print(selection)
            # selection should already be the correct size
            return selection, np.std(selection), np.min(selection), np.max(selection)

        lat_diff = int(lat2) - int(lat)
        lon_diff = int(lon2) - int(lon)

        # just get all 4 tiles even if only need 1 extra. Shouldn't happen often so don't need to worry about speed
        if lat_diff > 0 or lon_diff > 0:
            if lat_diff > 2 or lon_diff > 2:
                print("ERROR: Area exceeds 4 tiles--too large!")
                return None
            else:
                case = 4
                # get all for tiles, regardless of which ones we need..
                BL_tile, _, _, _ = read_elevation_from_zip_file(lat, lon, path=path)
                TL_tile, _, _, _ = read_elevation_from_zip_file(lat + 1, lon, path=path)
                BR_tile, _, _, _ = read_elevation_from_zip_file(lat, lon + 1, path=path)
                TR_tile, _, _, _ = read_elevation_from_zip_file(lat + 1, lon + 1, path=path)

                BL_tile = BL_tile[1:, :-1]  # trim 1 overlap on top and right row/column
                TL_tile = TL_tile[:, :-1]  # trim overlap on only right column
                BR_tile = BR_tile[1:, :]  # trim overlap on only top row
                # TR_tile no trim needed

                top_combined = np.concatenate((TL_tile, TR_tile), axis=1)
                bottom_combined = np.concatenate((BL_tile, BR_tile), axis=1)

                combined_tile = np.concatenate((top_combined, bottom_combined), axis=0)

                # start_row = int(abs(lat2-int(lat)) * (SAMPLES-1))
                # end_row =  SAMPLES * 2 + 1 - int(abs((lat-int(lat)) * (SAMPLES-1)))
                # start_col = SAMPLES * 2 - int(abs((lon-int(lon)) * (SAMPLES-1)))
                # end_col =  int(abs((lon2-int(lon)) * (SAMPLES-1)))

                # # works for NE hemisphere
                # start_row = SAMPLES * 2 + 1 - int((lat2-int(lat)) * (SAMPLES-1))
                # end_row =  SAMPLES * 2 + 1 - int(((lat-int(lat)) * (SAMPLES-1)))
                # start_col = int((lon - int(lon))*SAMPLES-1)
                # end_col =  int(((lon2-int(lon)) * (SAMPLES-1)))

                # # option 3
                # start_row = SAMPLES * 2 + 1 - int(abs(lat2-int(lat)) * (SAMPLES-1))
                # end_row =  SAMPLES * 2 + 1 - int(((lat-int(lat)) * (SAMPLES-1)))
                # start_col = int((lon - int(lon))*SAMPLES-1)
                # end_col =  int(((lon2-int(lon)) * (SAMPLES-1)))

                # option 4
                start_row = SAMPLES * num_tiles + 1 - int((lat2 - np.floor(lat)) * (SAMPLES - 1))
                end_row = SAMPLES * num_tiles + 1 - int((lat - np.floor(lat)) * (SAMPLES - 1))
                start_col = int((lon - np.floor(lon)) * (SAMPLES - 1))
                end_col = int((lon2 - np.floor(lon)) * (SAMPLES - 1))

                # start_row = 3000
                #end_row =  4000
                # start_col = 3000
                #end_col =  4500
                selection = combined_tile[start_row:end_row, start_col:end_col]

                #print(lat, lon, lat2, lon2)
                #print(BL_tile, TL_tile, BR_tile, TR_tile)
                #print(f"combined tile {combined_tile} with lengths {len(combined_tile)} {len(combined_tile[0])}")
                #print(start_row, end_row, start_col, end_col)
                #print(f"selection {selection}")
                # selection should already be the correct size
                # return combined_tile, np.std(combined_tile), np.min(combined_tile), np.max(combined_tile)

                return selection, np.std(selection), np.min(selection), np.max(selection)

    except Exception as e:
        print(f"Encountered an error {e} in read_exact_elevation_from_zip_file with"
              f" start/end: {start_row}/{end_row},{start_col}/{end_col} and selection {selection}")
        return None

    # # another more precise, but more complex way of doing it
    # if lat_diff > 0:
    #     if lat_diff > 2:
    #         print("ERROR: Area exceeds 4 tiles--too large!")
    #         return None
    #     if lon_diff > 0:
    #         if lon_diff > 2:
    #             print("ERROR: Area exceeds 4 tiles--too large!")
    #             return None
    #         # case 4 (need 4 tiles)
    #         # get_4_tiles()
    #         pass
    #     else:
    #         # case 2 (need 2 N-S tiles)
    #         # get_4_tiles()
    #         pass
    #
    # if lon_diff > 0:
    #     if lon_diff > 2:
    #         print("ERROR: Area exceeds 4 tiles--too large!")
    #         return None
    # else:
    #     # case 3 (need 2 E-W tiles)
    #     # get_4
    #     pass


def read_elevation_from_file(hgt_file, lat, lon):
    with open(hgt_file, 'rb') as hgt_data:
        # HGT is 16bit signed integer(i2) - big endian(>)
        elevations = np.fromfile(
            hgt_data,  # binary data
            np.dtype('>i2'),  # data type
            SAMPLES * SAMPLES  # length
        ).reshape((SAMPLES, SAMPLES))

        lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
        lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))

        return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)


def get_all_elev(hgt_file):
    with open(hgt_file, 'rb') as hgt_data:
        # HGT is 16bit signed integer(i2) - big endian(>)
        elevations = np.fromfile(
            hgt_data,  # binary data
            np.dtype('>i2'),  # data type
            SAMPLES * SAMPLES  # length
        ).reshape((SAMPLES, SAMPLES))

        return elevations


def get_std(hgt_file):
    with open(hgt_file, 'rb') as hgt_data:
        data = np.fromfile(
            hgt_data,  # binary data
            np.dtype('>i2'),  # data type
            SAMPLES * SAMPLES  # length
        )
        return np.std(data)


# reads city elevation standard deviation values and merged output, then creates a new file with the city,error,stdev
# really slow and inefficient but don't need to run this often so won't bother optimizing for now
# Fairland,US,39.07622,-76.95775,44.56766372846624 vs 2024_09_11-23_02_46-results.txt,Adeje__ES8.csv,1434.1417,4511

def save_elevationstdev_vs_error(elev_stdev_file='city_elev_stdev.csv',
                                 error_file='20240912_deny_list_merged_output.csv'):
    elev_stdev_data = pd.read_csv(elev_stdev_file)
    error_data = pd.read_csv(error_file)

    count = 0
    with open('20240912_error_vs_elev_stdev.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["city_id", "error", "stdev"]
        writer.writerow(header)
        for index, row in error_data.iterrows():
            if count % 1000 == 0:
                print(count)
            city_id = re.split(r'\d+', row['city_id'])[0]
            #print(city_id)
            for index2, row2 in elev_stdev_data.iterrows():
                try:
                    #print(row2)
                    city_id2 = row2['city'].replace(' ', '_') + '__' + row2['country']
                    #print(city_id2)
                    if city_id == city_id2:
                        #print(f"{city_id},{row['error']},{row2['stdev']}")
                        count += 1
                        #print(count)
                        row_to_write = [city_id, row['error'], row2['stdev']]
                        #print(row_to_write)
                        writer.writerow(row_to_write)

                except:
                    continue


# to generate city_elev_stdev.csv list
def write_city_elev_stdev():
    cities = read_city_list()
    results = generate_elev_stdev_list2(cities)
    #outfile_name = 'city_elev_stdev.csv'
    outfile_name = 'city_elev_stdev_exact.csv'
    with open(outfile_name, 'w', newline='', encoding="ISO-8859-1") as file:
        writer = csv.writer(file)
        header = ["geonameid","city","country", "lat", "lon", "stdev_elev", "min_elev", "max_elev"]
        writer.writerow(header)
        writer.writerows(results)


#save_elevationstdev_vs_error()

def test_and_plot_one_tile(lat=None, lon=None, path=SRTM_DIR, hgt_file='N40W112.hgt'):
    if lat != None and lon != None:
        elevation_data, _, _, _ = read_elevation_from_zip_file(lat, lon, path=path)
        hgt_file = get_hgt_filename(lat, lon)
    else:
        result = get_all_elev(hgt_file)

    # result2 = read_elevation_from_file(hgt_file, 40.67686418038837, -111.81474764770228)
    # result3, stdev, _, _ = read_elevation_from_zip_file(40.67686418038837, -111.81474764770228)
    # print(result)
    # print(f"len(results) = {len(result)} and len(result[0]) = {len(result[0])}")
    # print(f"result[0,0] result[0,-1] {result[0,0]} and {result[0,-1]}")
    # for row in result:
    #     print(row[1])
    # print(result2)
    # print(f"var of file is {get_std(hgt_file)}")
    # print(result3, stdev)
    # print(len(result3), len(result3[0]))
    #print(generate_elev_stdev_list(city_list))

    # # Open the .hgt file using rasterio
    # with rasterio.open(hgt_file) as dataset:
    #     # Read the data as a 2D array
    #     elevation_data = dataset.read(1)

    print(f"dimensions of elevation data tile: {len(elevation_data)},{len(elevation_data[0])}")

    # Create a simple plot of the elevation data
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_data, cmap='terrain', vmin=np.min(elevation_data), vmax=np.max(elevation_data))
    plt.colorbar(label='Elevation (m)')
    plt.title(f'Elevation Map for {hgt_file}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def test_and_plot_selection(lat, lon, lat2, lon2):
    elevation_data, stdev, _, _ = read_exact_elevation_from_zip_file(lat, lon, lat2, lon2)

    print(elevation_data)

    # Create a simple plot of the elevation data
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_data, cmap='terrain', vmin=np.min(elevation_data), vmax=np.max(elevation_data))
    plt.colorbar(label='Elevation (m)')
    plt.title(f'Elevation Map for {lat, lon, lat2, lon2}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


# NW test 40.67461689521767, -111.81602731110051,41.00440083289622, -111.44495208702567 (works)
# NE test 45.07356565385212, 36.029601330481846,45.42802393221889, 37.085084369227786 (works)
# SW test -43.19059492375988, -65.18856797121325,-42.18849806595958, -64.12039828411608 (works)
# SE test -5.266682797390478, 120.06496307062174,-4.081741580315065, 121.6523628870969 (works)
if __name__ == '__main__':
    #test_and_plot_one_tile(-12.51390150059538, 134.77031898051987)
    #test_and_plot_selection(-12.51390150059538, 134.77031898051987, -12.107735552459383, 135.86044588268203)
    write_city_elev_stdev()
