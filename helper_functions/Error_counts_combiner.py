import pandas as pd
import re
import csv
import numpy as np

# merges outputs for error results and sample count suitable for graphing


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
    with open('20241216_per_node_PL_with_bias-15000_cities.log', 'r') as f:
        lines = f.readlines()

    # Input long string
    data = lines[0]

    # Regular expression pattern to match the CSV filename and log10 values
    pattern = re.compile(
        r"(?P<filename>[^,]+\.csv),\{'None': \{'log10': (?P<log10>\d+\.\d+), 'log10_per_node': (?P<log10_per_node>\d+\.\d+)\}\}")

    # Extract and print results
    results_dict = {}
    for match in pattern.finditer(data):
        filename = match.group("filename").split("\\")[-1]  # Extract only the filename
        log10 = float(match.group("log10"))
        log10_per_node = float(match.group("log10_per_node"))
        results_dict[filename] = {"log10": log10, "log10_per_node": log10_per_node}

    # # Display results
    # for filename, log10, log10_per_node in results:
    #     print(f"Filename: {filename}, log10: {log10}, log10_per_node: {log10_per_node}")

    return results_dict

def combine_per_node_with_others(results_dict):
    input_file = "cities_error_and_counts.csv"
    with open(input_file, 'r',encoding = "ISO-8859-1") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        new_header = ["city_id", "CNN_error", "log10", "log10_per_node", "sample_counts"]

        na_counts = 0
        min_counts = {"CNN_error": 0, "log10": 0, "log10_per_node": 0}
        mins_list = []
        for row in reader:
            city_id = row[1]  # city_id column
            CNN_error = float(row[2])    # error column
            sample_counts = row[3]  # sample_counts column
            log10 = results_dict.get(city_id, {}).get("log10", "N/A")
            log10_per_node = results_dict.get(city_id, {}).get("log10_per_node", "N/A")
            print([city_id, CNN_error, log10, log10_per_node, sample_counts])



            if log10 == "N/A" or log10_per_node == "N/A":
                na_counts += 1
            else:
                log10 = float(log10)
                log10_per_node = float(log10_per_node)
                values = {"CNN_error": CNN_error, "log10": log10, "log10_per_node": log10_per_node}
                min_key = min(values, key=values.get)
                print(min_key)
                min_counts[min_key] += 1
                min_value = min_key
                mins_list.append(float(values[min_key]))
        print("Min counts:", min_counts)
        print("na_counts:", na_counts)
        print(f" average of min error {np.mean(np.array(mins_list))}")

    return


if __name__ == '__main__':
    #old_combiner()
    results = get_per_node_pl_data()
    print(results)
    combine_per_node_with_others(results)