from helium_training import main_process as run
from helium_training import get_rx_lats
import numpy as np
from helium_training import get_square_corners
import torch
import code
import glob


import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# collection of helper functions to run tests locally


# Move the function outside
def process_city(city):
    params = {'func_list': ["PATHLOSS"], "data_filename": city}
    try:
        results = run(params)
        return city, results
    except Exception as e:
        print(f"pathloss model failed for {city}: {e}")
        return city, None


def test_pathloss_on_cities_from_generated_multi(directory_path, log_file):
# multiprocess option to more quickly test
    csv_files = glob.glob(f"{directory_path}/*.csv")
    results_dict = {}
    total_count = len(csv_files)

    with ProcessPoolExecutor() as executor:
        future_to_city = {executor.submit(process_city, city): city for city in csv_files}
        completed_count = 0

        for future in as_completed(future_to_city):
            city = future_to_city[future]
            completed_count += 1

            try:
                city, results = future.result()
                if results is not None:
                    print(f"({completed_count}/{total_count}) {os.path.basename(city)} {results}")
                    results_dict[city] = results
                    with open(log_file, "a") as file:
                        file.write(f"{city},{results}\n")
                else:
                    print(f"Skipping {city} due to failure.")
            except Exception as e:
                print(f"Error processing {city}: {e}")

    return results_dict

def test_pathloss_on_cities_from_generated(directory_path,log_file):
    csv_files = glob.glob(f"{directory_path}/*.csv")
    results_dict = {}
    total_count = len(csv_files)
    count = 1
    # for city in csv_files:
    #     if "4235668" in city:
    #         print(f"found {city}")
    #         print(f"continuing from count {count}")
    #         break
    #     count += 1

    for city in csv_files[count:]:
        params = {'func_list': ["PATHLOSS"], "data_filename": city}
        try:
            results = run(params)
            tmp_city = city.split('\\')[-1]
            print(f"({count}/{total_count}) {tmp_city} {results}")
            results_dict[city] = results
            with open(log_file, "a") as file:
                file.write(f"{city},{results}\n")
        except Exception as e:
            print(f"pathloss model failed for {city} because {e}")
        count += 1
    return results_dict


def default_test():
    # rx_blacklist = ['47.50893434', '47.51250611', '47.61922143', '47.4692100571904', '47.47742889862316',
    #                      '47.47922825256919', '47.486512396667486', '47.50359742681539', '47.50840335356933',
    #                      '47.51072216107213', '47.51731330472538', '47.530655757870726', '47.53205816773291',
    #                      '47.53273388440102', '47.53393616926552', '47.538850220954174', '47.542722886879794',
    #                      '47.54327270368826', '47.54468061147836', '47.54580855981581', '47.55104682159288',
    #                      '47.553193946170445', '47.55472254606462', '47.56965033955307', '47.56985542720164',
    #                      '47.57011064833567', '47.57016078334357', '47.570418274093385', '47.57229596451633',
    #                      '47.61385761645424', '47.61531726651759', '47.61703782509865', '47.61707073349552',
    #                      '47.61771134123786', '47.61820102339988', '47.624933594863656', '47.62788464575477',
    #                      '47.6279178754742', '47.62792341916671', '47.62798810555178', '47.628999870304085',
    #                      '47.63317115265743', '47.63321182606039', '47.63370215464106', '47.63370706366582',
    #                      '47.63733751209776', '47.64694171272788', '47.65073198859608', '47.65390855096642',
    #                      '47.58049332880381', '47.58292549332933', '47.585471925347406', '47.58853890118896',
    #                      '47.58902400026363', '47.58944644884187', '47.59340963652312', '47.59864847808752',
    #                      '47.60566438533686', '47.60595473978473', '47.60637337907797', '47.609964745154365']


    # # 25th percentile remove one Seattle
    # rx_blacklist =['47.50893434', '47.51250611', '47.56667388', '47.61922143', '47.4692100571904', '47.47742889862316',
    #  '47.48925703694905', '47.50359742681539', '47.50840335356933', '47.508746888592974', '47.51072216107213',
    #  '47.51731330472538', '47.52369040459335', '47.530655757870726', '47.53205816773291', '47.53273388440102',
    #  '47.53393616926552', '47.53500568388223', '47.538850220954174', '47.54240398585348', '47.542722886879794',
    #  '47.54327270368826', '47.54468061147836', '47.54823440420244', '47.54837431128243', '47.553193946170445',
    #  '47.55351181613522', '47.554626222317054', '47.55472254606462', '47.55602653173352', '47.55647317987443',
    #  '47.559019618951936', '47.56965033955307', '47.56985542720164', '47.57011064833567', '47.57016078334357',
    #  '47.570418274093385', '47.57229596451633', '47.61067184190455', '47.61440590930977', '47.61536982933476',
    #  '47.61582542823447', '47.61707073349552', '47.61717326802679', '47.61749062748967', '47.61771134123786',
    #  '47.61820102339988', '47.624933594863656', '47.62788464575477', '47.6279178754742', '47.62798810555178',
    #  '47.62891578888903', '47.628999870304085', '47.63317115265743', '47.63321182606039', '47.63370215464106',
    #  '47.63370706366582', '47.63733751209776', '47.64030456039423', '47.64248991491972', '47.64694171272788',
    #  '47.65073198859608', '47.65390855096642', '47.57920936741801', '47.58019670193037', '47.58049332880381',
    #  '47.58292549332933', '47.585471925347406', '47.58853890118896', '47.58902400026363', '47.59340963652312',
    #  '47.594241262487685', '47.59864847808752', '47.60104020422019', '47.60566438533686', '47.60595473978473',
    #  '47.60637337907797', '47.60637979517027', '47.609964745154365']

    #10th percentile remove one Seattle
    rx_blacklist = ['47.50893434', '47.51250611', '47.61922143', '47.4692100571904', '47.47742889862316', '47.47922825256919',
     '47.486512396667486', '47.50359742681539', '47.50840335356933', '47.51072216107213', '47.51731330472538',
     '47.530655757870726', '47.53205816773291', '47.53273388440102', '47.53393616926552', '47.538850220954174',
     '47.542722886879794', '47.54327270368826', '47.54468061147836', '47.54580855981581', '47.55104682159288',
     '47.553193946170445', '47.55472254606462', '47.56965033955307', '47.56985542720164', '47.57011064833567',
     '47.57016078334357', '47.570418274093385', '47.57229596451633', '47.61385761645424', '47.61531726651759',
     '47.61703782509865', '47.61707073349552', '47.61771134123786', '47.61820102339988', '47.624933594863656',
     '47.62788464575477', '47.6279178754742', '47.62792341916671', '47.62798810555178', '47.628999870304085',
     '47.63317115265743', '47.63321182606039', '47.63370215464106', '47.63370706366582', '47.63733751209776',
     '47.64694171272788', '47.65073198859608', '47.65390855096642', '47.58049332880381', '47.58292549332933',
     '47.585471925347406', '47.58853890118896', '47.58902400026363', '47.58944644884187', '47.59340963652312',
     '47.59864847808752', '47.60566438533686', '47.60595473978473', '47.60637337907797', '47.609964745154365']


    datafile = r'C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240921_15000cities_normal\generated\5809844_Seattle__US8.csv'

    params = {"max_num_epochs":50,"num_training_repeats":1,"batch_size":64, "rx_blacklist":rx_blacklist,
              "coordinates" : np.array([(47.57027738863522, -122.38536489829036), (47.642142611364775, -122.27877510170964)]),
              'func_list':["MSE","EMD"],"data_filename":datafile}

    # params = {"max_num_epochs":1,"num_training_repeats":50,"batch_size":64,"rx_blacklist":rx_blacklist,
    #           'func_list':["EMD,MSE"],"data_filename":"datasets/helium_SD/filtered_Seattle_data.csv",
    #           "results_type":"default"}

    # params = {"max_num_epochs":50,"num_training_repeats":3,"batch_size":64,
    #           "coordinates" : np.array([(47.556372, -122.360229),(47.63509, -122.281609)]),
    #           'func_list':["COM","MSE","EMD"],"data_filename":"datasets/helium_SD/SEA30_driving2.csv"}

    # params = {"max_num_epochs": 50, "num_training_repeats": 3, "batch_size": 64,
    #           "coordinates": np.array([(47.556372, -122.360229), (47.63509, -122.281609)]),
    #           'func_list': ["COM"], "data_filename": "datasets/helium_SD/SEA30_helium.csv"}

    # La Jolla helium
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64,
    #           "coordinates": np.array([(32.868365149860296, -117.2512836119757),(32.91169449421575, -117.18643990160513)]),
    #           'func_list': ["COM"], "data_filename": "datasets/helium_SD/la-jolla_latlon_32.88_-117.23__8.csv"}

    # La Jolla driving
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64,
    #           "coordinates": np.array([(32.868365149860296, -117.2512836119757),(32.91169449421575, -117.18643990160513)]),
    #           'func_list': ["MSE"], "data_filename": "datasets/helium_SD/20250115_SD_drive_logging.csv"}

    # antwerp_file = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240921_15000cities_normal\generated\2803138_Antwerpen__BE8.csv"
    #
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64,
    #           "coordinates": np.array([(51.178,4.37), (51.250,4.45)]),
    #           'func_list': ["PATHLOSS"], "data_filename": antwerp_file}

    # antwerp_file = "datasets/antwerp_endpoint_dataset.csv"
    #
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64,
    #           "coordinates": np.array([(51.178,4.37), (51.250,4.45)]),
    #           'func_list': ["PATHLOSS"], "data_filename": antwerp_file}

    # antwerp_file = "datasets/antwerp_endpoint_dataset.csv"
    #
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64,
    #           "coordinates": np.array([(51.178,4.37), (51.250,4.45)]),
    #           'func_list': ["PATHLOSS"], "data_filename": antwerp_file}
    #
    # file = r'C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240926_top60euro_normal\generated\best_worst\3067696_Prague__CZ8.csv'
    #
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64,
    #           'func_list': ["PATHLOSS"], "data_filename": file}

    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE", "COM", "EMD"], "data_filename": "datasets/helium_SD/SD30_helium.csv",
    #           "results_type": "remove_one", "coordinates" : np.array([(32.732419, -117.247639), (32.796799, -117.160606)])}

    # params = {"max_num_epochs":50,"num_training_repeats":3,"batch_size":64,
    #           "coordinates" : np.array([(32.732419, -117.247639),(32.796799, -117.160606)]),
    #           'func_list':["COM","MSE","EMD"],"data_filename":"datasets/helium_SD/SD30_Fedex.csv"}

    # params = {"max_num_epochs": 10, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE", "COM"], "data_filename": "datasets/helium_SD/SF30_helium.csv",
    #           "timespan": (1675140329, 1675352459), "results_type": "remove_one",
    #           "coordinates": [(37.610424, -122.531204), (37.808156, -122.336884)]}

    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE", "COM", "EMD"], "data_filename": "datasets/helium_SD/SD30_driving.csv",
    #           "results_type": "remove_one", "coordinates": [(32.732419, -117.247639), (32.796799, -117.160606)]}

    # params = {"max_num_epochs": 1, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0], "splits": [],
    #           'func_list': ["MSE", "COM", "EMD"], "data_filename": "datasets/helium_SD/SEA30_helium.csv",
    #           "timespan": (1680610427, 1683238715),"results_type": "split_timespan_results",
    #           "coordinates": [(47.556372, -122.360229), (47.63509, -122.281609)]}
    #

    # params = {"max_num_epochs":50,"num_training_repeats":3,"batch_size":64,
    #           "coordinates" : np.array([(47.556372, -122.360229),(47.63509, -122.281609)]),
    #           'func_list':["COM","MSE","EMD"],"data_filename":"datasets/helium_SD/20240926_SEA_drive_logging.csv"}


    # TODO: is it filtering out the rest of the area and shrinking the size?? Need to investigate
    #
    # bl_coords, tr_coords = get_square_corners(32.881666, -117.233804, 8000)
    #
    # params = {"max_num_epochs": 50, "num_training_repeats": 1, "batch_size": 64, "rx_blacklist": [0],
    #           'func_list': ["MSE", "COM", "EMD"], "data_filename": "datasets/helium_SD/20241008_CSE_rooftop_data.csv",
    #           "results_type": "default", "coordinates": np.array([bl_coords, tr_coords])}

    # params =  {"max_num_epochs": 50, "num_training_repeats": 3, "batch_size": 64,
    #            "coordinates": np.array([(47.556372, -122.360229), (47.63509, -122.281609)]),
    #            'func_list': ["PATHLOSS"], "data_filename": "datasets/helium_SD/SEA30_helium.csv"}

    # params = {"max_num_epochs": 100, "num_training_repeats": 3, "batch_size": 64,
    #           "coordinates": np.array([(47.556372, -122.360229), (47.63509, -122.281609)]),
    #           'func_list': ["PATHLOSS"], "data_filename": "datasets/helium_SD/SEA30_driving2.csv"}

    # params = {"max_num_epochs": 50, "num_training_repeats": 3, "batch_size": 64,
    #           "coordinates": np.array([(47.556372, -122.360229), (47.63509, -122.281609)]),
    #           'func_list': ["PATHLOSS"], "data_filename": "datasets/helium_SD/SD30_helium.csv"}

    # params = {"max_num_epochs": 50, "num_training_repeats": 3, "batch_size": 64,
    #           "coordinates": np.array([(47.556372, -122.360229), (47.63509, -122.281609)]),
    #           'func_list': ["PATHLOSS"], "data_filename": "datasets/helium_SD/SD30_Fedex.csv"}

    rx_lats = get_rx_lats(params)
    print(rx_lats)

    results = run(params)

    print(f"printing results: {results}")

    # for key,item in results.items():
    #     for key2, item2 in item.items():
    #         print(key2,item2)#,item2.tx_vecs,item2.rx_vecs)
    #         print(item2.tx_vecs)

    # code.interact(local=locals())

    # for key in results.keys():
    #     min_err = [99999]
    #     for row in results[key]:
    #         if any("_test" in s for s in row) and float(row[-1] < min_err[-1]):
    #             min_err = row
    #     print(f"lowest error is row {min_err}")
    return(results)

if __name__ == '__main__':
    default_test()
    # directory_path = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240926_top60us_normal2\generated\best_worst"
    # results_dict_US = test_pathloss_on_cities_from_generated(directory_path,"Best_Worst_US_cities.log")
    #
    # directory_path = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240926_top60euro_normal\generated\best_worst"
    # results_dict_Euro = test_pathloss_on_cities_from_generated(directory_path,"Best_Worst_Euro_cities.log")

    # print(f"US cities best_worst results {results_dict_US.values()}")
    # print(f"Euro cities best_worst results {results_dict_Euro.values()}")

    # directory_path = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240921_15000cities_normal\generated"
    # results_dict_15000cities = test_pathloss_on_cities_from_generated_multi(directory_path,"15000_cities_v2_PL_MMSE.log")
    #
    # directory_path = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20250123_15000cities_v2\generated"
    # results_dict_15000cities = test_pathloss_on_cities_from_generated_multi(directory_path,"15000_cities_v2_PL_MMSE.log")


    #TODO: denylist filtered with pathloss
    #TODO: also need to repeat other correlation experiments on pathloss data..
    #TODO: also test on the same splits that the CNNs use, to make it "fair"?

    # print(results_dict)
    code.interact(local=locals())