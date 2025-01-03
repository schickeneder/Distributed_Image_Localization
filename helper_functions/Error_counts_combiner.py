import pandas as pd
import re
import csv
import numpy as np
import ast

# merges outputs for error results and sample count suitable for graphing
# combines and stores "city_id", "CNN_error", "log10", "log10_per_node", "sample_counts" for 11000+ cities


def old_combiner():
    # first_csv like: results_file,city_id,error
    # e.g.: 2024_09_08-07_29_24-results.txt,Ranst__BE8.csv,361.79547
    #first_csv = pd.read_csv('denylist_1500_cities.csv', encoding = "ISO-8859-1")
    first_csv = pd.read_csv('20240912_deny_list_1500cities_combined_results.csv', encoding = "ISO-8859-1")
    # second_csv like: city_id,sample_counts
    # e.g.: Caledon__CA8.csv,2
    #second_csv = pd.read_csv('denylist_1500_cities_sample_counts.csv', encoding = "ISO-8859-1")
    second_csv = pd.read_csv('20240912_denylist_15000_cities_sample_counts.csv', encoding = "ISO-8859-1")





    # Merge the two DataFrames on the 'city_id' column, using a left join to keep all rows from the first CSV
    merged_df = pd.merge(first_csv, second_csv[['city_id', 'sample_counts']], on='city_id', how='left')

    # Replace NaN values in the 'sample_counts' column with 0
    merged_df['sample_counts'].fillna(0, inplace=True)

    # Convert 'sample_counts' to integers if needed
    merged_df['sample_counts'] = merged_df['sample_counts'].astype(int)

    # Save the merged result back to a new CSV file
    #merged_df.to_csv('deny_list_merged_output.csv', index=False)
    merged_df.to_csv('20240912_deny_list_merged_output.csv', index=False)


    print("Data merged and saved to merged_output.csv")

def get_per_node_pl_data():
    # file containing log10 error with general pathloss and log10 per node pathloss with vector bias correction
    with open('20241216_per_node_PL_with_bias-15000_cities.log', 'r') as f:
        lines = f.readlines()

    # Input long string
    data = lines[0]

    # Regular expression pattern to match the CSV filename and log10 values
    pattern = re.compile(
        r"(?P<filename>[^,]+\.csv),\{'None': \{'log10': (?P<log10>\d+\.\d+), 'log10_per_node': (?P<log10_per_node>\d+\.\d+)\}\}")
    mag_bias_pattern = re.compile(r"(?P<filename>[^,]+\.csv),\{'None': \{'log10_per_node': \((?P<mag_bias>\d+\.\d+),")

    # Extract and print results
    results_dict = {}
    for match in pattern.finditer(data):
        filename = match.group("filename").split("\\")[-1]  # Extract only the filename
        log10 = float(match.group("log10"))
        log10_per_node = float(match.group("log10_per_node"))
        results_dict[filename] = {"log10": log10, "log10_per_node": log10_per_node}

    # Add mag_bias values from the second data source
    with open('20241223_vector_bias-error_mag-15000_cities.log', "r") as mag_bias_file:  # Replace with actual mag_bias file path
        for line in mag_bias_file:
            match = mag_bias_pattern.search(line)
            if match:
                filename = match.group("filename").split("\\")[-1]  # Extract only the filename
                mag_bias = float(match.group("mag_bias"))
                if filename in results_dict:
                    results_dict[filename]["mag_bias"] = mag_bias
                else:
                    results_dict[filename] = {"mag_bias": mag_bias}

    # # Display results
    # for filename, log10, log10_per_node in results:
    #     print(f"Filename: {filename}, log10: {log10}, log10_per_node: {log10_per_node}")

    # Now get MMSE results
    input_file = '20250103_PL_PnPL_MMSE_15000_cities.log'
    with open(input_file, 'r') as f:
        for line in f:
            tmp_line = line.strip().split('generated\\')[-1]
            filename, data = tmp_line.split('.csv,',1)
            filename += '.csv'
            # print(filename, data)
            data_dict = ast.literal_eval(data)
            if data_dict['None']:
                if filename in results_dict:
                    results_dict[filename]["log10_error_MSE"] = data_dict['None']['log10']
                    results_dict[filename]["log10_PN_error_MSE"] = data_dict['None']['log10_per_node'][1]
                    results_dict[filename]["log10_PN_mag_bias_MSE"] = data_dict['None']['log10_per_node'][0]
                else:
                    results_dict[filename] = {"log10_error_MSE" : data_dict['None']['log10'],
                                              "log10_PN_error_MSE": data_dict['None']['log10_per_node'][1],
                                              "log10_PN_mag_bias_MSE": data_dict['None']['log10_per_node'][0]}
                # print(results_dict[filename])

    log10_error_list = []
    log10_PN_list = []
    log10_error_MSE_list = []
    log10_PN_error_MSE_list = []

    for value in results_dict.values():
        print(value)
        try:
            log10_error_list.append(value['log10'])
            log10_PN_list.append(value['log10_per_node'])
            log10_error_MSE_list.append(value['log10_error_MSE'])
            log10_PN_error_MSE_list.append(value['log10_PN_error_MSE'])
        except:
            pass

    print(f"log10_error, log10_PN_error, log10_MSE_error, log10_PN_MSE_error {np.mean(log10_error_list)}, {np.mean(log10_PN_list)}, {np.mean(log10_error_MSE_list)}, {np.mean(log10_PN_error_MSE_list)}")
    print(f"counts: {len(log10_error_list)}, {len(log10_PN_list)}, {len(log10_error_MSE_list)}, {len(log10_PN_error_MSE_list)}")



    return results_dict

def get_nodes_stats():
    input_file = "20241226_distribution_stats-15000_cities.log"
    stats_dict = {}
    with open(input_file, 'r',encoding = "ISO-8859-1") as infile:
        for line in infile:
            # witness_counts = row['None']
            # print(witness_counts)
            city, tmp_row = line.split('.csv,',1)
            city = city.split('\\')[-1] + '.csv'
            processed_string = tmp_row.replace('array', 'np.array').replace(', dtype=int64', '')
            node_dict = eval(processed_string)
            try:
                witness_counts = list(node_dict['None']['stats'][0])
                grid_histogram = list(node_dict['None']['stats'][1])
                bins = list(node_dict['None']['stats'][2])
            except:
                continue

            mean_witness = np.mean(witness_counts)
            median_witness = np.median(witness_counts)
            if len(bins) > 1:
                percent_with_nodes = np.sum(grid_histogram[1:])/np.sum(grid_histogram)
            else:
                percent_with_nodes = 0
            if len(bins) > 2:
                percent_with_2_nodes = np.sum(grid_histogram[2:])/np.sum(grid_histogram)
            else:
                percent_with_2_nodes = 0

            # print(city, mean_witness, median_witness, percent_with_nodes, percent_with_2_nodes)
            # TODO: need percentage of grid spaces with at least 1 node, and " " " with at least 2 nodes for correlation
            # average and median number of witness events per transmitter
            # maybe also the difference in median vs average to use in correlation to show distribution change?

            stats_dict[city] = {"mean_witness": mean_witness, "median_witness": median_witness,
                                "percent_with_nodes": percent_with_nodes, "percent_with_2_nodes": percent_with_2_nodes}

    return stats_dict

def combine_per_node_with_others(results_dict,stats_dict=None, exclude_rows_missing_data=True):
    input_file = "cities_error_and_counts.csv"
    with open(input_file, 'r',encoding = "ISO-8859-1") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        new_header = ["city_id", "CNN_error", "log10", "log10_per_node", "mag_bias","sample_counts",
                      "mean_witness", "median_witness", "percent_with_nodes", "percent_with_2_nodes"]

        na_counts = 0
        min_counts = {"CNN_error": 0, "log10": 0, "log10_per_node": 0}
        mins_list = []
        data_list = []
        for row in reader:
            city_id = row[1]  # city_id column
            CNN_error = float(row[2])    # error column
            sample_counts = int(row[3])  # sample_counts column
            log10 = results_dict.get(city_id, {}).get("log10", "N/A")
            log10_per_node = results_dict.get(city_id, {}).get("log10_per_node", "N/A")
            mag_bias = results_dict.get(city_id, {}).get("mag_bias", "N/A")

            mean_witness = stats_dict.get(city_id, {}).get("mean_witness", "N/A")
            median_witness = stats_dict.get(city_id, {}).get("median_witness", "N/A")
            percent_with_nodes = stats_dict.get(city_id, {}).get("percent_with_nodes", "N/A")
            percent_with_2_nodes = stats_dict.get(city_id, {}).get("percent_with_2_nodes", "N/A")


            output_row = [city_id, CNN_error, log10, log10_per_node, mag_bias, sample_counts,
                          mean_witness, median_witness, percent_with_nodes, percent_with_2_nodes]
            # print(output_row)
            if exclude_rows_missing_data:
                if 'N/A' in output_row:
                    continue
                else:
                    data_list.append(output_row)


            if log10 == "N/A" or log10_per_node == "N/A":
                na_counts += 1
            else:
                log10 = float(log10)
                log10_per_node = float(log10_per_node)
                values = {"CNN_error": CNN_error, "log10": log10, "log10_per_node": log10_per_node}
                min_key = min(values, key=values.get)
                # print(min_key)
                min_counts[min_key] += 1
                min_value = min_key
                mins_list.append(float(values[min_key]))

        df = pd.DataFrame(data_list, columns=new_header)
        print(f"printing df {df}")
        numeric_df = df.select_dtypes(include=[float, int])
        print(f"printing numeric_df {numeric_df}")
        correlation_matrix = numeric_df.corr()
        print(f" mean of numeric_df {numeric_df.mean()}")
        print(f" std dev of numeric_df {numeric_df.std()}")
        print(f" zero count of numeric df per with nodes {(numeric_df["percent_with_nodes"] == 0.0).sum()}")
        print(f" zero count of numeric df per with 2 nodes {(numeric_df["percent_with_2_nodes"] == 0).sum()}")

        # # get correlation between 2nd and 4th column
        # data_array = np.array(data_list)
        # second_column = data_array[:, 3]
        # fifth_column = data_array[:, 5]
        # mask = (second_column != 'N/A') & (fifth_column != 'N/A')
        # filtered_second_column = second_column[mask].astype(float)
        # filtered_fourth_column = fifth_column[mask].astype(float)
        #
        # # Calculate correlation
        # correlation = np.corrcoef(filtered_second_column, filtered_fourth_column)[0, 1]
        #
        #
        # print("Correlation:", correlation)
        # data_array = np.array(data_list, dtype=float)[:,-2]
        # print(f"data_array {data_array} mean {data_array.mean()}")


        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        print(correlation_matrix)


        print("Min counts:", min_counts)
        print("na_counts:", na_counts)
        print(f" average of min error {np.mean(np.array(mins_list))}")

    return


if __name__ == '__main__':
    #old_combiner()

    stats_dict = get_nodes_stats()
    results = get_per_node_pl_data()
    # print(results)
    combine_per_node_with_others(results,stats_dict)
    print(f"length of results: {len(results)}")
