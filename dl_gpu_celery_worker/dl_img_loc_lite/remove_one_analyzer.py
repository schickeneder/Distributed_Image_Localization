import re
import numpy as np

""" Parse log file and capture test values for removed nodes

'removing row: 47.55779474389175-------------------------------------------' indicates new test with that node removed

These represent training test and validation errors for 
0.2testsize_train 3053.038
0.2testsize_test 3080.575
0.2testsize_train_val 3555.6702

We look for the node key (lat value in 'removing row') and then 'testsize_test #' and add that # to that node's dict

RESULTS for remove_one_log_filtered_Seattle-rnX1-3:

across all nodes
median com:5997.12765 mse:3055.809 emd:2992.9832
best model counts: [0, 1168, 1811]

25th percentile
['47.50893434', '47.51250611', '47.56667388', '47.61922143', '47.4692100571904', '47.47742889862316', '47.48925703694905', '47.50359742681539', '47.50840335356933', '47.508746888592974', '47.51072216107213', '47.51731330472538', '47.52369040459335', '47.530655757870726', '47.53205816773291', '47.53273388440102', '47.53393616926552', '47.53500568388223', '47.538850220954174', '47.54240398585348', '47.542722886879794', '47.54327270368826', '47.54468061147836', '47.54823440420244', '47.54837431128243', '47.553193946170445', '47.55351181613522', '47.554626222317054', '47.55472254606462', '47.55602653173352', '47.55647317987443', '47.559019618951936', '47.56965033955307', '47.56985542720164', '47.57011064833567', '47.57016078334357', '47.570418274093385', '47.57229596451633', '47.61067184190455', '47.61440590930977', '47.61536982933476', '47.61582542823447', '47.61707073349552', '47.61717326802679', '47.61749062748967', '47.61771134123786', '47.61820102339988', '47.624933594863656', '47.62788464575477', '47.6279178754742', '47.62798810555178', '47.62891578888903', '47.628999870304085', '47.63317115265743', '47.63321182606039', '47.63370215464106', '47.63370706366582', '47.63733751209776', '47.64030456039423', '47.64248991491972', '47.64694171272788', '47.65073198859608', '47.65390855096642', '47.57920936741801', '47.58019670193037', '47.58049332880381', '47.58292549332933', '47.585471925347406', '47.58853890118896', '47.58902400026363', '47.59340963652312', '47.594241262487685', '47.59864847808752', '47.60104020422019', '47.60566438533686', '47.60595473978473', '47.60637337907797', '47.60637979517027', '47.609964745154365']
10th percentile
['47.50893434', '47.51250611', '47.61922143', '47.4692100571904', '47.47742889862316', '47.47922825256919', '47.486512396667486', '47.50359742681539', '47.50840335356933', '47.51072216107213', '47.51731330472538', '47.530655757870726', '47.53205816773291', '47.53273388440102', '47.53393616926552', '47.538850220954174', '47.542722886879794', '47.54327270368826', '47.54468061147836', '47.54580855981581', '47.55104682159288', '47.553193946170445', '47.55472254606462', '47.56965033955307', '47.56985542720164', '47.57011064833567', '47.57016078334357', '47.570418274093385', '47.57229596451633', '47.61385761645424', '47.61531726651759', '47.61703782509865', '47.61707073349552', '47.61771134123786', '47.61820102339988', '47.624933594863656', '47.62788464575477', '47.6279178754742', '47.62792341916671', '47.62798810555178', '47.628999870304085', '47.63317115265743', '47.63321182606039', '47.63370215464106', '47.63370706366582', '47.63733751209776', '47.64694171272788', '47.65073198859608', '47.65390855096642', '47.58049332880381', '47.58292549332933', '47.585471925347406', '47.58853890118896', '47.58902400026363', '47.58944644884187', '47.59340963652312', '47.59864847808752', '47.60566438533686', '47.60595473978473', '47.60637337907797', '47.609964745154365']

"""
# infiles = ['run_log/remove_one_log_filtered_Seattle-rnX.txt', 'run_log/remove_one_log_filtered_Seattle-rnX2.txt',
#            'run_log/remove_one_log_filtered_Seattle-rnX3.txt',]

#infiles = ['run_log/remove_25th_p_filtered_Seattle.txt']
infiles = ['run_log/20240717_filtered_Seattle_all_helium.txt']



removing_row_pattern = re.compile(r"removing row:\s+([-+]?\d*\.\d+|\d+)") # re.compile(r"removing row:\s+([-+]?\d*\.\d+|\d+)")
testsize_test_pattern = re.compile(r"2testsize_test\s+([-+]?\d*\.\d+|\d+)") # this only matches random, not grid tests
model_pattern = re.compile(r'(.{3})\.pkl$')

remove_one_enabled = False # regular analysis vs remove-one dataset

def main():
    match_dict = {} # node: {com: [<values>], mse: [<values>], emd: [<values>]}
    full_list = [] # list of all values across all nodes all models
    full_lists_model = {"com": [], "mse": [], "emd": []} # contains a key for each model with a list of all values all nodes for that model within
    lines = []
    for infile_path in infiles:
        with open(infile_path) as f:
            # ----------------------- from dataset < 5
            lines += f.readlines()
            #print(lines)
    current_node = None
    current_model = None
    if remove_one_enabled == False:
        current_node = "all"
        match_dict[current_node] = {}

    for line in lines:
        match_node = removing_row_pattern.search(line)
        match_test = testsize_test_pattern.search(line)
        match_model = model_pattern.search(line)

        if remove_one_enabled:
            if match_node:
                current_node = match_node.group(1)
                if current_node not in match_dict:
                    match_dict[current_node] = {}
                #print(match_node.group(1))

        if match_model:
            current_model = match_model.group(1)
            #match_dict[current_node][current_model] = []

        if match_test: # any test results will belong to current node until new node is found via "removing row.."
            #print(line)
            #print(match_test.group(1))
            result = float(match_test.group(1))
            if current_model not in match_dict[current_node]:
                match_dict[current_node][current_model] = []
            match_dict[current_node][current_model].append(result)
            full_list.append(result)
            full_lists_model[current_model].append(result)

    for key in match_dict.keys():
        print(f"Node {key}")
        for model in match_dict[key].keys():
            print(f"Model {model}")
            print(min(match_dict[key][model]), max(match_dict[key][model]), np.percentile(np.array(match_dict[key][model]),50))#, match_dict[key][model])
        # this is to see which model performs the best in order of COM MSE EMD..
        # print(f"min index {match_dict[key].index(min(match_dict[key]))}, max index {match_dict[key].index(max(match_dict[key]))}")
    # print(min(full_list), max(full_list), statistics.median(full_list))

    # print percentiles
    dict_percentiles = {}
    for model in ["com","mse","emd"]:
        print(f"{model}:",end='')
        dict_percentiles[model] = {}
        for percentile in [.001,0.01,0.5,1,5,10,25,50,75,90]:
            value = np.percentile(np.array(full_lists_model[model]),percentile)
            dict_percentiles[model][percentile] = value
            print(f"{value:.2f} ",end='')
        print("\n")
    #print(f"median com:{np.percentile(np.array(full_lists_model["com"]),50)} mse:{np.percentile(np.array(full_lists_model["mse"]),50)} emd:{np.percentile(np.array(full_lists_model["emd"]),50)}")

    best_model_counts = [0,0,0] # represent com, mse and emd
    for node in match_dict.keys():
        for com,mse,emd in zip(match_dict[node]["com"],match_dict[node]["mse"],match_dict[node]["emd"]):
            winner = min(com,mse,emd)
            best_model_counts[[com,mse,emd].index(winner)] += 1
    print(f"best model counts: {best_model_counts}")

    # We want to get a list of nodes that, when removed, reduce the error;
    # if for either MSE or EMD model, it reduces below threshold, include it. COM model didn't perform well so who cares about it
    nodes_below_75 = []
    nodes_below_50 = []
    nodes_below_25 = []
    nodes_below_10 = []
    nodes_below_05 = []
    for key in match_dict.keys():
        for model in ["mse","emd"]:
            if np.percentile(np.array(match_dict[key][model]),75) < dict_percentiles[model][75]:
                if key not in nodes_below_75:
                    nodes_below_75.append(key)
            if np.percentile(np.array(match_dict[key][model]),50) < dict_percentiles[model][50]:
                if key not in nodes_below_50:
                    nodes_below_50.append(key)
            if np.percentile(np.array(match_dict[key][model]),25) < dict_percentiles[model][25]:
                if key not in nodes_below_25:
                    nodes_below_25.append(key)
            if np.percentile(np.array(match_dict[key][model]),10) < dict_percentiles[model][10]:
                if key not in nodes_below_10:
                    nodes_below_10.append(key)
            if np.percentile(np.array(match_dict[key][model]),0.5) < dict_percentiles[model][0.5]:
                if key not in nodes_below_05:
                    nodes_below_05.append(key)


    print(nodes_below_75)
    print(nodes_below_50)
    print(nodes_below_25)
    print(nodes_below_10)
    print(f"Nodes < 75: {len(nodes_below_75)} < 50: {len(nodes_below_50)} < 25: {len(nodes_below_25)} < 10: {len(nodes_below_25)} < 0.5: {len(nodes_below_05)} of {len(match_dict)} nodes")



            # print(min(match_dict[key][model]), max(match_dict[key][model]), np.percentile(np.array(match_dict[key][model]),50))




    # # to get sorted list of nodes
    # tmp_list = [float(x) for x in match_dict.keys()]
    # print(tmp_list)
    # print(f"sorted list {sorted(tmp_list)}")




if __name__ == '__main__':
    main()
