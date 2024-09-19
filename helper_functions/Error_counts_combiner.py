import pandas as pd

# merges outputs for error results and sample count suitable for graphing

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
