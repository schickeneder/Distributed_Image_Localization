import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import zipfile
import csv

SRTM_DICT = {'SRTM1': 3601, 'SRTM3': 1201}

# Get the type of SRTM files or use SRTM3 by default
SRTM_TYPE = os.getenv('SRTM_TYPE', 'SRTM3')
SAMPLES = SRTM_DICT[SRTM_TYPE]

SRTM_DIR = r"C:\Users\ps\Helium_Data\SRTM Elevation Data\SRTMGL1_003-20240911_232824"

city_list = r"cities15000extracted_citycountry_lat_lon.txt"

RAW_CITIES = r"cities15000_raw.txt"

# read raw city list from ....?
# create list of dictionaries for each city
def read_city_list(file_path=RAW_CITIES):
    cities = []
    with open(file_path, mode='r', encoding='utf-8') as file:
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
        results = read_elevation_from_zip_file(lat,lon)
        if results:
            stdev = results[1]
            new_list.append([cityname, country, lat,lon, stdev])
        count += 1
    return new_list



def read_elevation_from_zip_file(lat, lon, path=SRTM_DIR):
    if lat > 0:
        filenamepart1 = 'N' + f"{abs(int(lat)):02d}"
    else:
        filenamepart1 = 'S' + f"{abs(int(lat)):02d}" # filename is always SW corner
    if lon > 0:
        filenamepart2 = 'E' + f"{abs(int(lon)):03d}"
    else:
        filenamepart2 = 'W' + f"{abs(int(lon)-1):03d}"  # filename is always SW corner

    target_filename = filenamepart1 + filenamepart2

    for filename in os.listdir(path):
        if target_filename in filename and filename.endswith(".zip"):
            zip_path = os.path.join(SRTM_DIR, filename)
            break

    target_filename += ".hgt"
    #print(target_filename)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(target_filename) as hgt_data:
                with open('tmp', 'wb') as output_file:
                    output_file.write(hgt_data.read())
                elevations = np.fromfile(
                    'tmp',  # binary data
                    np.dtype('>i2'),  # data type
                    SAMPLES * SAMPLES  # length
                ).reshape((SAMPLES, SAMPLES))

        return elevations, np.std(elevations)


    except Exception as e:
        print(f"Couldn't find zip file {target_filename} because {e}")
        return None


    # with open(hgt_file, 'rb') as hgt_data:
    #     # HGT is 16bit signed integer(i2) - big endian(>)
    #     elevations = np.fromfile(
    #         hgt_data,  # binary data
    #         np.dtype('>i2'),  # data type
    #         SAMPLES * SAMPLES  # length
    #     ).reshape((SAMPLES, SAMPLES))
    #
    #     lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
    #     lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))
    #
    #     return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)

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

hgt_file = 'N40W112.hgt'
result = get_all_elev(hgt_file)

result2 = read_elevation_from_file(hgt_file,40.67686418038837, -111.81474764770228)
result3,stdev = read_elevation_from_zip_file(40.67686418038837, -111.81474764770228)
# print(result)
# print(f"len(results) = {len(result)} and len(result[0]) = {len(result[0])}")
# print(f"result[0,0] result[0,-1] {result[0,0]} and {result[0,-1]}")
# for row in result:
#     print(row[1])
# print(result2)
# print(f"var of file is {get_std(hgt_file)}")
print(result3,stdev)
#print(generate_elev_stdev_list(city_list))

cities = read_city_list()
results = generate_elev_stdev_list(cities)
outfile_name = 'city_elev_stdev.csv'
with open(outfile_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)




# Open the .hgt file using rasterio
with rasterio.open(hgt_file) as dataset:
    # Read the data as a 2D array
    elevation_data = dataset.read(1)

# Create a simple plot of the elevation data
plt.figure(figsize=(10, 8))
plt.imshow(elevation_data, cmap='terrain', vmin=np.min(elevation_data), vmax=np.max(elevation_data))
plt.colorbar(label='Elevation (m)')
plt.title(f'Elevation Map for {hgt_file}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()