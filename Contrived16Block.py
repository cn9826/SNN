import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math


#%%
## Define the categories of stimulus
def shapeMatrix(category, dimension=6, high_pix=1, low_pix=0, num_corrode=0):
    category_list = ["X", "O", "<<"]
    if not category in category_list:
        print("Error when calling shapeMatrix: category {} is not defined!".format(category))
        exit(1)
    matrix = np.full((dimension, dimension), fill_value=low_pix, dtype=int, order='C')
    # f_slash and b_slash occupy a quater of dimension x dimension matrix
    f_slash = np.full((dimension/2, dimension/2), fill_value=low_pix, dtype=int, order='C')
    b_slash = np.full((dimension/2, dimension/2), fill_value=low_pix, dtype=int, order='C')

    if category == "O":
        matrix[0, np.array([1,2])] = high_pix
        matrix[1, np.array([0,3])] = high_pix
        matrix[2, np.array([0,3])] = high_pix
        matrix[3, np.array([1,2])] = high_pix
    elif category == "X":
        matrix[0, np.array([0,3])] = high_pix
        matrix[1, np.array([1,2])] = high_pix
        matrix[2, np.array([1,2])] = high_pix
        matrix[3, np.array([0,3])] = high_pix
    elif category == "<<":
        matrix[0, np.array([1,3])] = high_pix
        matrix[1, np.array([0,2])] = high_pix
        matrix[2, np.array([0,2])] = high_pix
        matrix[3, np.array([1,3])] = high_pix
    if num_corrode != 0:
        random_indices =  [
                            (random.randint(0,dimension-1), random.randint(0,dimension-1))
                            for i in range(num_corrode)
                        ]
        for index in random_indices:
            matrix[index] = low_pix
    
    return matrix
def imshowMatrix (fig_idx, matrix, cmap='gray'):
    extent=(0, matrix.shape[1], matrix.shape[0], 0)
    fig = plt.figure(fig_idx)
    ax1=plt.subplot(1,1,1)
    ax1.imshow(matrix, extent=extent, cmap=cmap)
    ax1.set_xticks(np.arange(0,matrix.shape[1],1))
    ax1.set_yticks(np.arange(matrix.shape[0]-1,-1,-1))
    ax1.hlines(y=range(1,matrix.shape[1]), xmin=0, xmax=4, lw=2, color='0.5', linestyles='dashed')
    ax1.hlines(y=matrix.shape[1]/2, xmin=0, xmax=4, lw=4, color='0.1', linestyles='solid')

    ax1.vlines(x=range(1,matrix.shape[0]), ymin=0, ymax=4, lw=2, color='0.5', linestyles='dashed')
    ax1.vlines(x=matrix.shape[0]/2, ymin=0, ymax=4, lw=4, color='0.1', linestyles='solid')

def index_duplicate (seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def index_2d (list_2d, element):
    for row, row_list in enumerate(list_2d):
        if element in row_list:
            return (row, row_list.index(element))

def plotNeuronResponse (sn):
    t = np.arange(0, sn.duration, sn.dt)
    fig = plt.figure(sn.neuron_idx)
    fig.subplots_adjust(hspace=0.5)

    hax1 = plt.subplot(2, 1 ,1)
    plt.plot(t, sn.u, '-', lw=2)
    plt.grid(True)
    hax1.set(title="Synaptic Current" + r"$u(t)$" + " of neuron {}".format(sn.neuron_idx))

    hax2 = plt.subplot(2, 1, 2)
    plt.plot(t, sn.v, '-', lw=2)
    
    plt.grid(True)
    hax2.set(title="Membrane Potential" + r"$v(t)$" + " of neuron {}".format(sn.neuron_idx),
             xlabel="Timestep"
            )
    hax2.hlines(y=sn.threshold, xmin=t[0], xmax=t[-1], lw=2, color='0.3', linestyles='dashed')
    
    # fig.show()
    
def plotNeuronResponse_iterative(sn_list, epochs_list, instance_list, only_output_layer=1):
    for epoch in epochs_list:
        for instance in instance_list:            
            if only_output_layer == 0:
                for i in range(len(sn_list[epoch][instance])):
                    plotNeuronResponse(sn_list[epoch][instance][i])
            else:
                for i in range(len(sn_list[epoch][instance])):
                    if sn_list[epoch][instance][i].layer_idx == num_layers - 1:
                        plotNeuronResponse(sn_list[epoch][instance][i])

def randomInt(mean, std, num):
    value_array = np.random.normal(mean,std,num)
    value_list = [int(value) for value in value_array]
    return value_list

#%% Parameters to tune
######################################################################################
printout_dir = "sim_printouts/Contrived16Block/"

## Specify Global Connectivity Parmeters
num_neurons_perLayer = [8,3]                           # Assuming num_neurons_perLayer is the number of connections in FC case
max_num_fires = 1

## Specify common Spiking Neuron Parameters
duration = 200
tau_u = 14      # in units with respect to duration
tau_v = None     # in units with respect to duration
vth_low = 2
vth_high = 120

## Supervised Training Parameters
supervised_training_on = 1      # turn on/off supervised training 
separation_window = 16
stop_num = 20
coarse_fine_ratio=0.5

## Training Dataset Parameters
num_instances =100              # number of training instances per epoch

## Simulation Settings
debug_mode = 1
plot_response = 0

if supervised_training_on:
    printout_dir = printout_dir + "Supervised/BRRC/dumpsim.txt"
else:
    printout_dir = printout_dir + "Inference/BRRC/dumpsim.txt"
f_handle = open(printout_dir, "w+")

######################################################################################

#%% Generate Input & Output Patterns also checking dimensions
######################################################################################
## Define Input & Output Patterns
early_in = 0
late_in = 16

weight_vector = \
    [
        10, 10, 10, 10, 10, 10, 10, 10,

    ]

O_in = [late_in] * num_neurons_perLayer[0]
X_in = [late_in] * num_neurons_perLayer[0]
A_in = [late_in] * num_neurons_perLayer[0]
for i in range(num_neurons_perLayer[0]):
    if i == 0:
        O_in[i] = early_in
        A_in[i] = early_in
    elif i == 1:
        X_in[i] = early_in
        A_in[i] = early_in
    elif i == 2:
        X_in[i] = early_in
    elif i == 3:
        O_in[i] = early_in
    elif i == 4:
        X_in[i] = early_in
    elif i == 5:
        O_in[i] = early_in
    elif i == 6:
        O_in[i] = early_in
        A_in[i] = early_in
    elif i == 7:
        X_in[i] = early_in
        A_in[i] = early_in

input_pattern = \
    {
        "O"     :   O_in,
        "X"     :   X_in,        
        "<<"    :   A_in
    }
output_pattern = \
    {
        "O"     :   sum(num_neurons_perLayer[0:-1]),
        "X"     :   sum(num_neurons_perLayer[0:-1]) + 1,
        "<<"     :   sum(num_neurons_perLayer[0:-1]) + 2
    }

## Create stimulus spikes at the inuput layer (layer 0)
# Dimension: num_instances x num_input_neurons
stimulus_time_vector = [
                            {
                                "in_pattern"    :    None,
                                "in_latency"    :    [None] * num_neurons_perLayer[0]
                            }
                            for instance in range(num_instances)
                       ]

for instance in range(num_instances):
    stimulus_time_vector[instance]["in_pattern"] = \
            random.choice(list(input_pattern.keys()))
    stimulus_time_vector["in_latency"] = \
input_pattern[stimulus_time_vector[instance]["in_pattern"]]


## Specify the index of the desired output layer neuron to fire
desired_ff_neuron = [
                        {
                            "in_pattern"            :   None,
                            "out_pattern"           :   None,
                            "ff_neuron"             :   None
                        } for instance in range(num_instances)
                    ]

for instance in range(num_instances):
    desired_ff_neuron[instance]["in_pattern"] = \
        stimulus_time_vector[instance]["in_pattern"]
    desired_ff_neuron[instance]["out_pattern"] = \
        stimulus_time_vector[instance]["out_pattern"]
    desired_ff_neuron[instance]["ff_neuron"] = \
        output_pattern[desired_ff_neuron[instance]["out_pattern"]]


## Check dimensions
if len(stimulus_time_vector) != num_instances:
    print("Error: Dimension of stimulus does not match the number of instances per epoch")
    exit(1)
else:
    if len(stimulus_time_vector[0]["in_latency"]) != num_neurons_perLayer[0]:
        print("Error: Dimension of stimulus does not match the number of input neurons")
        exit(1)

if len(desired_ff_neuron) != num_instances:
    print("Error: Dimension of desired output time vector does not match the number of instances per epoch")
    exit(1)
######################################################################################

#%% Initialization before simulation loop
######################################################################################
stimulus_vector_info =  [
                            [] 
                            for instance in range(num_instances)
                        ] 
for instance in range(num_instances):
    for synapse_idx in range(num_neurons_perLayer[0]):
        stimulus_entry = {}
        stimulus_entry["fan_in_synapse_addr"] = synapse_idx
        stimulus_entry["time"] = stimulus_time_vector[instance]["in_latency"][synapse_idx]
        stimulus_vector_info[instance].append(stimulus_entry)

num_layers = len(num_neurons_perLayer)
num_neurons = sum(num_neurons_perLayer)
num_synapses = 0
for layer in range(num_layers):
    if layer == 0:                                      # layer 0 is the input layer, only having 1 connection
        num_synapses += num_neurons_perLayer[0]
    else:
        num_synapses += num_neurons_perLayer[layer] * num_neurons_perLayer[layer-1]


## Initialize neuron indices and synpase address
neuron_indices = range(num_neurons)
synapse_addr = range(num_synapses)

## Initialize Connectivity Table and its "fan_in_synapse_idx", "fan_out_synapse_idx" attributes 
ConnectivityTable = SNN.ConnectivityInfo(num_neurons=num_neurons, 
                                         max_num_connections=max(num_neurons_perLayer), 
                                         max_num_fires=max_num_fires
                                        )
last_layer_last_synapse = -1
for layer in range (num_layers):
    for neuron in range(num_neurons_perLayer[layer]):
        if layer == 0:
            TableIdx = neuron
            ConnectivityTable.fan_in_synapse_addr[TableIdx][0] = synapse_addr[TableIdx]      # input layer neurons have fan-in of 1  
        else:
            temp_list = synapse_addr[last_layer_last_synapse + 1 + neuron*num_neurons_perLayer[layer-1] : 
                                    last_layer_last_synapse + 1 + (neuron+1)*num_neurons_perLayer[layer-1]]
            TableIdx = neuron + sum(num_neurons_perLayer[0:layer])                   # non-input layer neurons have fan-in of num_neurons_perLayer[layer-1]
            ConnectivityTable.fan_in_synapse_addr[TableIdx][0:num_neurons_perLayer[layer-1]] = temp_list
            # ConnectivityTable.fan_out_synapse_addr[sum(num_neurons_perLayer[0:layer])-num_neurons_perLayer[layer-1] : sum(num_neurons_perLayer[0:layer])][neuron] =  
            for i in range(num_neurons_perLayer[layer-1]):
                ConnectivityTable.fan_out_synapse_addr[sum(num_neurons_perLayer[0:layer])-num_neurons_perLayer[layer-1]+i][neuron] = temp_list[i]     
        ConnectivityTable.layer_num[TableIdx] = layer
    
    # update last_layer_last_synapse
    if layer == 0:
        last_layer_last_synapse = synapse_addr[last_layer_last_synapse + (neuron+1)*1]
    else:
        last_layer_last_synapse = synapse_addr[last_layer_last_synapse + (neuron+1)*num_neurons_perLayer[layer-1]]

