import training_example
import csv

infile_path = 'datasets/helium_SD/filtered_Seattle_data.csv'
outfile_path = 'datasets/helium_SD/remove_one_tmp.csv'

# expecting format [count/time, txlat, txlon, rxlat, rxlon, ...]
def get_unique_RX_lats():
    rx_lats = []
    with open(infile_path) as f:
        # ----------------------- from dataset < 5
        lines = f.readlines()
    for line in lines[1:]: # skip headers
        columns = line.strip().split(',')
        rx_lat = float(columns[3])
        if rx_lat not in rx_lats:
            rx_lats.append(rx_lat)
    return rx_lats

def remove_RXer(lat):
    with open(infile_path, mode='r',newline='') as infile, open(outfile_path, mode='w',newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read the header row
        header = next(reader)
        # Write the header row to the output file
        writer.writerow(header)


        for row in reader:
            if float(row[3]) != float(lat):
                writer.writerow(row)
            else:
                #print(f"removing row: {row}")
                continue

def main():
    #rx_lats = get_unique_RX_lats()
    rx_lats = [47.61067184190455, 47.61274896248364, 47.61385761645424, 47.61407163295049, 47.61440590930977, 47.61531726651759, 47.61536982933476, 47.61582542823447, 47.61601732997836, 47.61693276293637, 47.61703782509865, 47.61707073349552, 47.61717326802679, 47.61749062748967, 47.617695863407725, 47.61771134123786, 47.61820102339988, 47.61922143, 47.62057503186795, 47.620995388945936, 47.622227627702976, 47.622737797478045, 47.62322122175047, 47.624933594863656, 47.62686456409219, 47.62788464575477, 47.6279178754742, 47.62792341916671, 47.62798810555178, 47.62843255293026, 47.62891578888903, 47.628999870304085, 47.62912786890652, 47.63069888487882, 47.63076599705895, 47.63239987, 47.63317115265743, 47.63321182606039, 47.63370215464106, 47.63370706366582, 47.63452284297265, 47.635165199804646, 47.635339209548185, 47.63732453863744, 47.63733751209776, 47.637900637507286, 47.63861293515679, 47.64028075699959, 47.64030456039423, 47.64115559957341, 47.64248991491972, 47.64486621071836, 47.64694171272788, 47.65073198859608, 47.65390855096642, 47.65459566364993]

    for lat in rx_lats:
        print(f"removing row: {lat}-------------------------------------------")
        remove_RXer(lat)
        training_example.main()

if __name__ == '__main__':
    main()