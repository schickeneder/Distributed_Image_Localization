import csv
import math

# compares results between denylist error outputs on raw error outputs from model

def read_csv(filepath):
    data = {}
    with open(filepath, mode='r', newline='', encoding='ISO-8859-1') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            # Assuming the CSV structure is [col1, col2, col3]
            col2 = row[1]  # 2nd column (index 1)
            col3 = float(row[2])  # 3rd column (index 2), convert to float
            data[col2] = col3
    return data


def compare_csv(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    print(f"Matching rows from column 2 and their column 3 differences:")
    differences = []

    for key in data1:
        if key in data2:
            diff = data1[key] - data2[key]
            differences.append(diff)

            print(f"{key},{diff}")

    if differences:
        avg_diff = sum(differences) / len(differences)

        # Calculate standard deviation
        variance = sum((x - avg_diff) ** 2 for x in differences) / len(differences)
        std_dev = math.sqrt(variance)

        print(f"\nAverage Difference = {avg_diff}")
        print(f"Standard Deviation = {std_dev}")
    else:
        print("\nNo matching rows found.")


if __name__ == "__main__":
    file1 = 'deny_list_merged_output.csv'
    file2 = 'merged_output.csv'

    compare_csv(file1, file2)
