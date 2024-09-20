import csv
import pickle

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

    cities, cities_data = read_city_list(file_path)

    # print(city_dict)

    square_length = 8000

    for city in cities_data:
        lat, lon = float(cities_data[city]['latitude']), float(cities_data[city]['longitude'])
        data_filename = "datasets/generated/" + cities_data[city]['geonameid'] + "_" + cities_data[city]["name"] \
                        + "__" + cities_data[city]["country"] + str(int(square_length / 1000)) + '.csv'

        print(city,lat,lon,data_filename)

    pickle.dump(city_dict, open('cities15000_dict_all.pickle', 'wb'))

    # Example: Filter cities by country code 'AU' for Australia
    au_cities = filter_cities_by_country(cities, 'AU')

    filtered_city_list_result = filtered_city_list(cities)
    for city in filtered_city_list_result:
        print(f'("{city[0]}, {city[1]}",{city[2]},{city[3]}),')
    # Print the city information
    #print_city_info(filtered_cities)


if __name__ == '__main__':
    main()
