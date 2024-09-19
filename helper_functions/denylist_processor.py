import pickle
import h3
import csv
import pandas as pd

# generates a denylist by node latitude for with with rx_blacklist parameters in model

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def find_matches(strings_list, list_of_lists):
    results = []
    strings_list_len = len(strings_list)
    count = 0

    for string in strings_list:
        if count % 100 == 0:
            print(f"Count: {count}/{strings_list_len}")
        for sublist in list_of_lists:
            if string == sublist[0]:
                # Append the 4th element (index 3) to the results list
                results.append(sublist[3])
                break  # Stop searching after finding the match
        count += 1
    return results


def find_matches_dict(strings_list, list_of_lists):
    # Create a dictionary for quick lookup
    lookup_dict = {lst[0]: lst[3] for lst in list_of_lists}

    return [lookup_dict[string] for string in strings_list if string in lookup_dict]

def build_lists():
    denylist = load_pickle('unique_denylist.pickle')
    node_locations = load_pickle('locations.pickle')

    h3_deny_list = find_matches_dict(denylist, node_locations)

    coords_deny_list = []
    for h3_index in h3_deny_list:
        coords_deny_list.append(h3.h3_to_geo(h3_index))

    return coords_deny_list,h3_deny_list


def get_deny_lat_list():
    coords_deny_list, *_ = build_lists()
    return [value[0] for value in coords_deny_list]

try:
    with open('deny_lat_list.csv', mode='r', newline='') as file:
        reader = csv.reader(file)

        # Read the single row
        result = next(reader)
    # df = pd.read_csv('deny_lat_list.csv')
    # result = df.iloc

except Exception as e:
    print(f"Couldn't read file: {e}")
    result = get_deny_lat_list()
    with open('deny_lat_list.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)


print(result)
print(len(result))