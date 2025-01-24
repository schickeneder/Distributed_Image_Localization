import csv
import pickle
import pandas as pd
import os
import math
import code
# citylist source files obtained from https://download.geonames.org/export/dump/

# returns a list and dict of cities with geonameid as dict key
def read_city_list(file_path):
    city_list = []
    city_dict = {}
    with open(file_path, mode='r', encoding = "ISO-8859-1") as file:
        reader = csv.reader(file, delimiter='\t')  # Use tab delimiter
        next(reader)  # Skip header if present
        for row in reader:
            if len(row) >= 15:  # Ensure there are enough columns
                city = {
                    'geonameid': row[0],
                    'name': row[1],
                    #'alternate_names': row[2],
                    'latitude': row[4],
                    'longitude': row[5],
                    'country': row[8],
                    'population': row[-5],
                    'timezone': row[-2],
                }
                city_list.append(city)
                city_dict[row[0]] = city
    return city_list, city_dict


def filter_cities_by_country(cities, country_code):
    return [city for city in cities if city['country'] == country_code]


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

def filtered_city_list(cities):
    filtered_cities = []
    filter_city_list = '20240912_deny_list_1500cities_combined_results.csv'
    with open(filter_city_list, mode='r', encoding = "ISO-8859-1") as file:
        reader = csv.reader(file)
        for row in reader:
            for city in cities:
                city['name'] = city['name'].replace(' ','_')
                target = row[1].split('__')[0]
                #print(f"row {row} comparing city {city['name']} to target {target}")

                if target in city['name']:
                    # print(f"row {row} comparing city {city['name']} to target {target}")
                    # print("Match!")
                    filtered_cities.append([city["name"], city["country"], city["latitude"], city["longitude"]])
    return filtered_cities



def main():
    file_path = 'cities15000_raw.txt' # from geonames.org, contains info for all cities with pop > 15000

    # list, dict
    cities, cities_data = read_city_list(file_path)

    # print(city_dict)

    square_length = 8000

    for city in cities_data:
        lat, lon = float(cities_data[city]['latitude']), float(cities_data[city]['longitude'])
        data_filename = "datasets/generated/" + cities_data[city]['geonameid'] + "_" + cities_data[city]["name"] \
                        + "__" + cities_data[city]["country"] + str(int(square_length / 1000)) + '.csv'

        print(city,lat,lon,data_filename)

    pickle.dump(cities_data, open('cities15000_dict_all.pickle', 'wb'))

    # Example: Filter cities by country code 'AU' for Australia
    au_cities = filter_cities_by_country(cities, 'AU')

    filtered_city_list_result = filtered_city_list(cities)
    for city in filtered_city_list_result:
        print(f'("{city[0]}, {city[1]}",{city[2]},{city[3]}),')
    # Print the city information
    #print_city_info(filtered_cities)

def generate_datasets(cities_data):

    square_length = 8000 # 8000 meters ~ 5 miles

    # ("Bourzanga, BF", 13.67806, -1.54611)


    # cities_data dict like '4366476': {'geonameid': '4366476', 'name': 'Randallstown', 'latitude': '39.36733', 'longitude': '-76.79525', 'country': 'US', 'population': '32430', 'timezone': 'America/New_York'}
    # cities_data = pickle.load(open('datasets/cities15000_dict_all.pickle', 'rb'))
    cities_data = pickle.load(open(cities_data, 'rb'))



    if global_dataset_loaded:
        print("Building datasets")
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

# geofilter global dataset to produce regional subset
def filter_coordinates(df, coordinates):
    bottom_left, top_right = (coordinates)
    return df[
        (df['lat1'] >= bottom_left[0]) & (df['lat1'] <= top_right[0]) &
        (df['lon1'] >= bottom_left[1]) & (df['lon1'] <= top_right[1]) &
        (df['lat2'] >= bottom_left[0]) & (df['lat2'] <= top_right[0]) &
        (df['lon2'] >= bottom_left[1]) & (df['lon2'] <= top_right[1])
    ]

if __name__ == '__main__':
    # main()
    global_dataset_loaded = False # flag updates when global_dataset load completes
    global_dataset = None
    load_global_dataset(r"C:\Users\ps\Helium_Data\all_nz_v2_di9.csv")
    code.interact(local=locals())
    generate_datasets("cities15000_dict_all.pickle")

# Errors with the following cities due to characters in their names when running this on Windows
# Couldn't save local dataset datasets/generated/12156875_Wexford/Maryvale__CA8.csv because Cannot save file into a non-existent directory: 'datasets\generated\12156875_Wexford'
# Couldn't save local dataset datasets/generated/2658656_ZÃ¼rich (Kreis 11) / Seebach__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\2658656_ZÃ¼rich (Kreis 11) '
# Couldn't save local dataset datasets/generated/2659310_ZÃ¼rich (Kreis 11) / Oerlikon__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\2659310_ZÃ¼rich (Kreis 11) '
# Couldn't save local dataset datasets/generated/2660306_ZÃ¼rich (Kreis 10) / HÃ¶ngg__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\2660306_ZÃ¼rich (Kreis 10) '
# Couldn't save local dataset datasets/generated/2661513_Biel/Bienne__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\2661513_Biel'
# Couldn't save local dataset datasets/generated/2661666_ZÃ¼rich (Kreis 4) / Aussersihl__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\2661666_ZÃ¼rich (Kreis 4) '
# Couldn't save local dataset datasets/generated/6295475_ZÃ¼rich (Kreis 10) / Wipkingen__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295475_ZÃ¼rich (Kreis 10) '
# Couldn't save local dataset datasets/generated/6295484_ZÃ¼rich (Kreis 11) / Affoltern__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295484_ZÃ¼rich (Kreis 11) '
# Couldn't save local dataset datasets/generated/6295495_ZÃ¼rich (Kreis 2) / Wollishofen__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295495_ZÃ¼rich (Kreis 2) '
# Couldn't save local dataset datasets/generated/6295498_ZÃ¼rich (Kreis 3) / Sihlfeld__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295498_ZÃ¼rich (Kreis 3) '
# Couldn't save local dataset datasets/generated/6295504_ZÃ¼rich (Kreis 6) / Unterstrass__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295504_ZÃ¼rich (Kreis 6) '
# Couldn't save local dataset datasets/generated/6295512_ZÃ¼rich (Kreis 9) / Albisrieden__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295512_ZÃ¼rich (Kreis 9) '
# Couldn't save local dataset datasets/generated/6295513_ZÃ¼rich (Kreis 9) / Altstetten__CH8.csv because Cannot save file into a non-existent directory: 'datasets\generated\6295513_ZÃ¼rich (Kreis 9) '
# Couldn't save local dataset datasets/generated/2849156_Reichenbach/Vogtland__DE8.csv because Cannot save file into a non-existent directory: 'datasets\generated\2849156_Reichenbach'
# Couldn't save local dataset datasets/generated/8593855_Neustadt/Nord__DE8.csv because Cannot save file into a non-existent directory: 'datasets\generated\8593855_Neustadt'
# Couldn't save local dataset datasets/generated/8593856_Neustadt/SÃ¼d__DE8.csv because Cannot save file into a non-existent directory: 'datasets\generated\8593856_Neustadt'
# Couldn't save local dataset datasets/generated/3104499_Gasteiz / Vitoria__ES8.csv because Cannot save file into a non-existent directory: 'datasets\generated\3104499_Gasteiz '
# Couldn't save local dataset datasets/generated/3110044_Donostia / San SebastiÃ¡n__ES8.csv because Cannot save file into a non-existent directory: 'datasets\generated\3110044_Donostia '
# Couldn't save local dataset datasets/generated/3116503_Arrasate / MondragÃ³n__ES8.csv because Cannot save file into a non-existent directory: 'datasets\generated\3116503_Arrasate '
# Couldn't save local dataset datasets/generated/3118228_Laudio / Llodio__ES8.csv because Cannot save file into a non-existent directory: 'datasets\generated\3118228_Laudio '
# Couldn't save local dataset datasets/generated/12718709_Kamuhanda / Ruyenzi__RW8.csv because Cannot save file into a non-existent directory: 'datasets\generated\12718709_Kamuhanda '