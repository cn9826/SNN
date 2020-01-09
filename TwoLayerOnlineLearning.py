import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
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
    
def plotNeuronResponse_iterative(sn_list, neuron_list, instance_list):
    for instance in instance_list:
        for neuron_idx in neuron_list:
            plotNeuronResponse(sn_list[instance][neuron_idx])

def randomInt(mean, std, num):
    value_array = np.random.normal(mean,std,num)
    value_list = [int(value) for value in value_array]
    return value_list

def BimodalLatency(latency_mode, mean_early, std_early, mean_late, std_late, low_lim=0, high_lim=64):
    latency_mode_list = ["early", "late"]
    if not latency_mode in latency_mode_list:
        print("Error when calling BimodalLatency: illegal specification of \"latency\"")
        exit(1)
    if latency_mode == "early":
        latency = np.random.normal(mean_early, std_early)
        if latency < low_lim:
            latency = low_lim
    elif latency_mode == "late":
        latency = np.random.normal(mean_late, std_late)
        if latency > high_lim:
            latency = high_lim
    return int(latency)
    
def getInLatencies(in_pattern, num_in_neurons, early_latency_list, late_latency_list,
                    mean_early, std_early, mean_late, std_late, low_lim=0, high_lim=64):

    in_pattern_list = ["O", "X", "<<", "//", ">>", "UA", "DA", "BS", "Bad//"]
    if not in_pattern in in_pattern_list:
        print("Error when calling getInLatencies: illegal specification of \"in_pattern\"")
        exit(1)
    
    InLatencies = [None] * num_in_neurons
    for i in range(num_in_neurons):
        latency_late = \
            BimodalLatency("late", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
        InLatencies[i] = latency_late
        late_latency_list.append(latency_late)

    if in_pattern == "O":
        for i in [0, 3, 5, 6]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == "X":
        for i in [1, 2, 4, 7]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == "<<":
        for i in [0, 1, 6, 7]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == "//":
        for i in [0, 1, 2, 3]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == ">>":
        for i in [2, 3, 4, 5]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == "UA":
        for i in [0, 2, 5, 7]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == "DA":
        for i in [1, 3, 4, 6]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    elif in_pattern == "BS":
        for i in [4, 5, 6, 7]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)
    elif in_pattern == "Bad//":
        for i in [0, 1, 2, 6]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[i] = latency_early
            early_latency_list.append(latency_early)

    return (InLatencies, early_latency_list, late_latency_list)

def plotInLatencyDistribution(early_latency_list, late_latency_list, tau_u, num_bins=8):
    fig, ax = plt.subplots(figsize=(14,7))
    early_list_tau = [latency / tau_u for latency in early_latency_list]
    late_list_tau = [latency / tau_u for latency in late_latency_list]
    ax.hist([early_list_tau, late_list_tau], bins=num_bins, label=['early latency','late latency'],
            edgecolor='k')

    ax.set_title('Stimulus Spike Latency Distribution in units of ' + r'$\tau_u$'
                ,fontsize=18, fontweight='bold')
    ax.axvline(x=num_bins/2, linewidth=2, color='k', linestyle='dashed')
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(num_bins/2, max_ylim*0.8, r'early-late cutoff ${}\tau_u$'.format(int(num_bins/2)),
            fontsize=12)
    ax.set_xticks(range(0, num_bins+1))
    xticklabel_list = [r'{}$\tau_u$'.format(i) for i in range(0, num_bins+1)]
    ax.set_xticklabels(xticklabel_list)
    ax.legend(fontsize=15)

#%% Parameters to tune
######################################################################################
printout_dir = "sim_printouts/Contrived16Block2Layer/"

## Specify Global Connectivity Parmeters
num_neurons_perLayer = [8, 12, 4]       # Assuming num_neurons_perLayer is the number of connections in FC case
num_connect_perNeuron = [1,4,-1]        # -1 denotes FC       

num_in_spikes_hidden = 2
num_in_spikes_output = 4

max_num_fires = 1

fan_in_neuron = [
                    [], [], [], [], [], [], [], [],
                    [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], 
                    [2, 3, 6, 7], [2, 3, 6, 7], [2, 3, 6, 7], [2, 3, 6, 7], [2, 3, 6, 7], [2, 3, 6, 7],
                    [x for x in range(8, 20)], 
                    [x for x in range(8, 20)], 
                    [x for x in range(8, 20)], 
                    [x for x in range(8, 20)]
                ]

initial_weight_hidden = [5] * num_connect_perNeuron[1] * num_neurons_perLayer[1] 
initial_weight_output = [5] * num_neurons_perLayer[-2] * num_neurons_perLayer[-1]
weight_vector = \
    [
        10, 10, 10, 10, 10, 10, 10, 10,
        *initial_weight_hidden,
        *initial_weight_output
    ]

## Specify common Spiking Neuron Parameters
duration = 150
tau_u = 8      # in units with respect to duration
tau_v = None     # in units with respect to duration
vth_input = 1
vth_hidden = 40 + 16     # with 2-spike consideration: [(2-1) x 5 x tau_u, 2 x 5 x tau_u)
                         # with 2-spike consideration: [(2-1) x 7 x tau_u, 2 x 7 x tau_u)

vth_output = 150        # with 3-spike consideration: [(4-1) x 5 x tau_u, 4 x 5 x tau_u)  
                         # with 3-spike consideration: [(4-1) x 7 x tau_u, 4 x 7 x tau_u)  
## Supervised Training Parameters
supervised_hidden = 1      # turn on/off supervised training in hidden layer
supervised_output = 1      # turn on/off supervised training in output layer 
separation_window = 10
stop_num = 50
coarse_fine_ratio=0.2

## Training Dataset Parameters
num_instances = 4000             # number of training instances per epoch

## Simulation Settings
debug_mode = 1
plot_response = 0
plot_InLatency = 0

if supervised_hidden or supervised_output:
    printout_dir = printout_dir + "Supervised/dumpsim.txt"
else:
    printout_dir = printout_dir + "Inference/dumpsim.txt"
f_handle = open(printout_dir, "w+")
f_handle.write("supervised_hidden: {}\n".format(supervised_hidden))
f_handle.write("supervised_output: {}\n".format(supervised_output))

######################################################################################
#%% Generate Input & Output Patterns also checking dimensions
######################################################################################
## Define Input & Output Patterns
mean_early = 0*2*tau_u + 2.5*tau_u
std_early = int(5*tau_u/3)
mean_late = 4*2*tau_u - 2.5*tau_u
std_late = int(5*tau_u/3)


input_patterns = ("O", "X", "UA", "DA")

output_pattern = \
    {
        "O"      :   sum(num_neurons_perLayer[0:-1]),
        "X"      :   sum(num_neurons_perLayer[0:-1]) + 1,
        "UA"     :   sum(num_neurons_perLayer[0:-1]) + 2,
        "DA"     :   sum(num_neurons_perLayer[0:-1]) + 3
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

early_latency_list = []
late_latency_list = []
for instance in range(num_instances):
    stimulus_time_vector[instance]["in_pattern"] = \
            random.choice(input_patterns)
    stimulus_time_vector[instance]["in_latency"], early_latency_list, late_latency_list = \
            getInLatencies(in_pattern=stimulus_time_vector[instance]["in_pattern"],
                           num_in_neurons=num_neurons_perLayer[0],
                           early_latency_list=early_latency_list,
                           late_latency_list=late_latency_list,
                           mean_early=mean_early, std_early=std_early,
                           mean_late=mean_late, std_late=std_late,
                           low_lim=0, high_lim=mean_late+2*tau_u  
                           )


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
        stimulus_time_vector[instance]["in_pattern"]
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
#%% Connectivity Initialization
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
    if num_connect_perNeuron[layer] != -1:
        num_synapses += num_neurons_perLayer[layer] * num_connect_perNeuron[layer]
    else:
        num_synapses += num_neurons_perLayer[layer] * num_neurons_perLayer[layer-1]

## check fan_in_neuron specifications
if len(fan_in_neuron) != num_neurons:
    print("num_neurons {} does not correspond to len(fan_in_neuron) {}!"
    .format(num_neurons, len(fan_in_neuron)))
    exit(1)
else:
    for layer in range(num_layers):
        for neuronInLayer in range(num_neurons_perLayer[layer]):
            if layer == 0:
                neuron_idx = neuronInLayer
                if len(fan_in_neuron[neuron_idx]) != 0:
                    print("Inappropriate specification in fan_in_neuron[{}]: input layer neuron should not have fan-in neurons!"
                    .format(neuron_idx))
                    exit(1)
            else:
                neuron_idx = neuronInLayer + sum(num_neurons_perLayer[0:layer])
                if num_connect_perNeuron[layer] != -1:  # Not FC
                    if len(fan_in_neuron[neuron_idx]) != num_connect_perNeuron[layer]:
                        print("Inappropriate specification in fan_in_neuron[{}]: len(fan_in_neuron[{}]) {} does not correspond to the specifed num_connect_perNeuron[{}] {}!"
                        .format(neuron_idx, neuron_idx, len(fan_in_neuron[neuron_idx]), layer, num_connect_perNeuron[layer]))
                        exit(1)
                else:                                   # FC
                    if len(fan_in_neuron[neuron_idx]) != num_neurons_perLayer[layer-1]:
                        print("Inappropriate specification in fan_in_neuron[{}]: len(fan_in_neuron[{}]) {} FC does not correspond to the specifed num_neurons_perLayer[{}] {}!"
                        .format(neuron_idx, neuron_idx, len(fan_in_neuron[neuron_idx]), layer, num_neurons_perLayer[layer]))
                        exit(1)


## Initialize neuron indices and synpase address
neuron_indices = range(num_neurons)
synapse_addrs = range(num_synapses)


## Initialize Connectivity Table and its "fan_in_synapse_idx", "fan_out_synapse_idx" attributes 
ConnectivityTable = SNN.ConnectivityInfo(num_neurons=num_neurons)

last_synapse = [None for i in range(num_layers)]
# assign fan_in_neuron_idx and fan_in_synapse_addr
for layer in range(num_layers):
    for neuron in range(num_neurons_perLayer[layer]):
        if layer == 0:
            TableIdx = neuron
            ConnectivityTable.layer_num[TableIdx] = layer
            ConnectivityTable.fan_in_synapse_addr[TableIdx].append(synapse_addrs[TableIdx])      # input layer neurons have fan-in of 1  
            last_synapse[layer] = num_neurons_perLayer[0] - 1
      
        else:
            TableIdx = neuron + sum(num_neurons_perLayer[0:layer])
            ConnectivityTable.layer_num[TableIdx] = layer
            # assign fan_in_neuron_idx
            ConnectivityTable.fan_in_neuron_idx[TableIdx] = fan_in_neuron[TableIdx]
            # assign fan_in_synapse_addr 
            if num_connect_perNeuron[layer] != -1:            
                ConnectivityTable.fan_in_synapse_addr[TableIdx] = \
                    [synapse_idx for synapse_idx in 
                        range(neuron * num_connect_perNeuron[layer] + last_synapse[layer-1] + 1,
                              (neuron+1) * num_connect_perNeuron[layer] + last_synapse[layer-1] + 1
                             )
                    ]
                if neuron == num_neurons_perLayer[layer] - 1:
                    last_synapse[layer] = (neuron+1) * num_connect_perNeuron[layer] + last_synapse[layer-1]
                    
            elif num_connect_perNeuron[layer] == -1:
                ConnectivityTable.fan_in_synapse_addr[TableIdx] = \
                    [synapse_idx for synapse_idx in 
                        range(neuron * num_neurons_perLayer[layer-1] + last_synapse[layer-1] + 1,
                              (neuron+1) * num_neurons_perLayer[layer-1] + last_synapse[layer-1] + 1
                             )
                    ]
                if neuron == num_neurons_perLayer[layer] - 1:
                    last_synapse[layer] = (neuron+1) * num_neurons_perLayer[layer-1] + last_synapse[layer-1] + 1

# assign fan_out_neuron_idx and fan_out_synapse_addr
for layer in range(num_layers):
    for neuron in range(num_neurons_perLayer[layer]):
        if layer == 0:
            TableIdx = neuron
            for n in range(num_neurons_perLayer[0], sum(num_neurons_perLayer[0:2])):
                if ((TableIdx in fan_in_neuron[n]) and 
                (not n in ConnectivityTable.fan_out_neuron_idx[TableIdx])):
                    ConnectivityTable.fan_out_neuron_idx[TableIdx].append(n)
                    if num_connect_perNeuron[layer+1] != -1:
                        ConnectivityTable.fan_out_synapse_addr[TableIdx].append(
                            (n-sum(num_neurons_perLayer[0:layer+1]))*num_connect_perNeuron[layer+1] + \
                                last_synapse[layer]+ 1 + ConnectivityTable.num_fan_in_neurons_established[n]
                        )
                    elif num_connect_perNeuron[layer+1] == -1:
                        ConnectivityTable.fan_out_synapse_addr[TableIdx].append(
                            (n-sum(num_neurons_perLayer[0:layer+1]))*num_neurons_perLayer[layer] + \
                                last_synapse[layer] + 1 + ConnectivityTable.num_fan_in_neurons_established[n]
                        )
                    ConnectivityTable.num_fan_in_neurons_established[n] += 1
        else:
            TableIdx = neuron + sum(num_neurons_perLayer[0:layer])
            # scan the fan-in neurons of the neurons in the next layer
            for n in range(sum(num_neurons_perLayer[0:layer+1]), sum(num_neurons_perLayer[0:layer+2])):
                if ((TableIdx in fan_in_neuron[n]) and 
                (not n in ConnectivityTable.fan_out_neuron_idx[TableIdx])):
                    ConnectivityTable.fan_out_neuron_idx[TableIdx].append(n)
                    if num_connect_perNeuron[layer+1] != -1:
                        ConnectivityTable.fan_out_synapse_addr[TableIdx].append(
                            (n-sum(num_neurons_perLayer[0:layer+1]))*num_connect_perNeuron[layer+1] + \
                                last_synapse[layer] + 1 + ConnectivityTable.num_fan_in_neurons_established[n]
                        )
                
                    elif num_connect_perNeuron[layer+1] == -1:
                        ConnectivityTable.fan_out_synapse_addr[TableIdx].append(
                            (n-sum(num_neurons_perLayer[0:layer+1]))*num_neurons_perLayer[layer] + \
                                last_synapse[layer] + 1 + ConnectivityTable.num_fan_in_neurons_established[n]
                        )
                    ConnectivityTable.num_fan_in_neurons_established[n] += 1

## Initialize WeightRAM
WeightRAM = SNN.WeightRAM(num_synapses)
for synapse_addr in synapse_addrs:
    post_neuron_idx_WRAM, post_connection_num = index_2d(ConnectivityTable.fan_in_synapse_addr, synapse_addr)
    WeightRAM.post_neuron_idx[synapse_addr] = post_neuron_idx_WRAM
    WeightRAM.weight[synapse_addr] = weight_vector[synapse_addr]
    if synapse_addr >= num_neurons_perLayer[0]:    
        pre_neuron_idx_WRAM, pre_connection_num = index_2d(ConnectivityTable.fan_out_synapse_addr, synapse_addr)
        WeightRAM.pre_neuron_idx[synapse_addr] = pre_neuron_idx_WRAM



## Initialize PotentialRAM 
PotentialRAM = SNN.PotentialRAM(num_neurons=num_neurons, num_instances=num_instances)
PotentialRAM.fan_out_synapse_addr = ConnectivityTable.fan_out_synapse_addr


## Initialize a list of all SpikingNeuron objects
sn_list = \
    [
        [] for instance in range(num_instances)
    ]
for instance in range(num_instances):
    for i in neuron_indices:
        layer_idx = ConnectivityTable.layer_num[i]
        if layer_idx == 0:        
            sn = SNN.SpikingNeuron( layer_idx=layer_idx,
                                    neuron_idx=i, 
                                    fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[i],
                                    fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[i],
                                    depth_causal = 1,
                                    depth_anticausal = 0,
                                    tau_u=tau_u,
                                    tau_v=tau_v,
                                    threshold=vth_input,
                                    duration=duration,
                                    max_num_fires=max_num_fires,
                                    training_on=0,
                                    supervised=0
                                    )
        if layer_idx == 1:
            depth_causal = 2
            depth_anticausal = 2
            if supervised_hidden:
                training_on = 1
                supervised = 1
            sn = SNN.SpikingNeuron( layer_idx=layer_idx,
                                    neuron_idx=i, 
                                    fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[i],
                                    fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[i],
                                    depth_causal = depth_causal,
                                    depth_anticausal = depth_anticausal,
                                    tau_u=tau_u,
                                    tau_v=tau_v,
                                    threshold=vth_hidden,
                                    duration=duration,
                                    max_num_fires=max_num_fires,
                                    training_on=training_on,
                                    supervised=supervised
                                    )

        if layer_idx == 2:
            depth_causal = 4
            depth_anticausal = 4
            if supervised_hidden:
                training_on = 1
                supervised = 1
            sn = SNN.SpikingNeuron( layer_idx=layer_idx,
                                    neuron_idx=i, 
                                    fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[i],
                                    fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[i],
                                    depth_causal = depth_causal,
                                    depth_anticausal = depth_anticausal,
                                    tau_u=tau_u,
                                    tau_v=tau_v,
                                    threshold=vth_output,
                                    duration=duration,
                                    max_num_fires=max_num_fires,
                                    training_on=training_on,
                                    supervised=supervised
                                    )
        
        sn_list[instance].append(sn)

#%% Inter-neuron data Initialization
######################################################################################
fired_synapse_list =    [
                            [
                                [] for step in range(0, sn.duration, sn.dt)
                            ]
                            for instance in range(num_instances)
                        ]   # num_instances x num_timesteps x num_fired_synapses

fired_neuron_list =    [
                            [
                                [] for step in range(0, sn.duration, sn.dt)
                            ]
                            for instance in range(num_instances)
                        ]   # num_instances x num_timesteps x num_fired_neurons

spike_info = [
                [
                    {    
                        "fired_synapse_addr": [],
                        "time"              : None
                    }
                    for step in range(0, sn.duration, sn.dt)
                ] for instance in range(num_instances)
             ]            
             # a list of dictionaries sorted by time steps

## Initialize statistics
hidden_neuron_fire_info =   [
                                {
                                    "neuron_idx":   [],
                                    "time"      :   []
                                } 
                                for instance in range(num_instances)
                            ]   

output_neuron_fire_info =   [
                                {
                                    "neuron_idx":   [],
                                    "time"      :   []
                                } 
                                for instance in range(num_instances)
                            ]   

inference_correct =     [    
                            None for instance in range(num_instances) 
                        ]   # num_instances

#%% Simulation Loop
######################################################################################
correct_cnt = 0
max_correct_cnt = 0
PreSynapticIdx_intended = \
    [
        {
            "causal"        :   None,
            "anti-causal"   :   None
        } for i in range(num_instances)
    ]

PreSynapticIdx_nonintended = \
    [
        {
            "causal"        :   None,
            "anti-causal"   :   None
        } for i in range(num_instances)
    ]
    
## Loop instances
for instance in range(num_instances):
    f_handle.write("---------------Instance {} {} -----------------\n".format(instance,stimulus_time_vector[instance]["in_pattern"]))
    
    ## Forward Pass
    for sim_point in range(0, sn.duration, sn.dt):
        # first check if any input synpase fires at this time step
        if sim_point in stimulus_time_vector[instance]["in_latency"]:
            fired_synapse_list[instance][sim_point].extend(
                index_duplicate(stimulus_time_vector[instance]["in_latency"], sim_point)
            )
            spike_info[instance][sim_point]["fired_synapse_addr"].extend(
                index_duplicate(stimulus_time_vector[instance]["in_latency"], sim_point)
            )

        spike_info[instance][sim_point]["time"] = sim_point
        
        for i in neuron_indices:
            sn_list[instance][i].accumulate(sim_point=sim_point, 
                                    spike_in_info=spike_info[instance][sim_point], 
                                    WeightRAM_inst=WeightRAM,
                                    debug_mode=debug_mode,
                                    instance=instance,
                                    f_handle=f_handle
                                    )                                            
            # upadate the current potential to PotentialRAM
            PotentialRAM.potential[instance][i] = sn_list[instance][i].v[sim_point]
            
            # update the list of synapses that fired at this sim_point
            if (sn_list[instance][i].fire_cnt != -1):
                if (sn_list[instance][i].spike_out_info[sn_list[instance][i].fire_cnt]["time"] == sim_point):
                    if (len(sn_list[instance][i].fan_out_synapse_addr) == 1): # if single fan-out
                        fired_synapse_list[instance][sim_point].append(val)
                        spike_info[instance][sim_point]["fired_synapse_addr"].append(val)
                    else:   # if multiple fan-out synpases
                        for key,val in enumerate(sn_list[instance][i].fan_out_synapse_addr):
                            fired_synapse_list[instance][sim_point].append(val)
                            spike_info[instance][sim_point]["fired_synapse_addr"].append(val)
                    
                    fired_neuron_list[instance][sim_point].append(sn_list[instance][i].neuron_idx)

                    # if the fired neuron at this sim_point is in the hidden layer
                    if (sn_list[instance][i].layer_idx == num_layers - 2):
                        hidden_neuron_fire_info[instance]["neuron_idx"].append(sn_list[instance][i].neuron_idx)
                        hidden_neuron_fire_info[instance]["time"].append(sim_point)

                    # if the fired neuron at this sim_point is in the output layer
                    if (sn_list[instance][i].layer_idx == num_layers - 1):
                        output_neuron_fire_info[instance]["neuron_idx"].append(sn_list[instance][i].neuron_idx)
                        output_neuron_fire_info[instance]["time"].append(sim_point)
    ## End of one Forward Pass

    # At the end of the Forward Pass, inspect output-layer firing info
    if len(output_neuron_fire_info[instance]["neuron_idx"]) > 0:
        # find the minimum of firing time and its corresponding list index
        min_fire_time = min(output_neuron_fire_info[instance]["time"])
        list_min_idx = index_duplicate(output_neuron_fire_info[instance]["time"], min_fire_time)
        f2f_neuron_lst =    [
                                output_neuron_fire_info[instance]["neuron_idx"][list_idx]
                                for list_idx in list_min_idx
                            ] 
        if desired_ff_neuron[instance]["ff_neuron"] in f2f_neuron_lst:
            f2f_neuron_idx = desired_ff_neuron[instance]["ff_neuron"]
        else:
            f2f_neuron_idx = f2f_neuron_lst[0]

        # find the non-F2F neurons that fired within separation window
        non_f2f_neuron_lst = [  
                                neuron_idx
                                for list_idx, neuron_idx in enumerate(output_neuron_fire_info[instance]["neuron_idx"])
                                if output_neuron_fire_info[instance]["time"][list_idx] - min_fire_time >= 0
                                and output_neuron_fire_info[instance]["time"][list_idx] - min_fire_time <= separation_window
                                and neuron_idx != f2f_neuron_idx
                                ]
    
    else:
        min_fire_time = None
        list_min_idx = None
        f2f_neuron_lst = []
        f2f_neuron_idx = None
        non_f2f_neuron_lst = []
    
    correct_cnt = SNN.combined_RSTDP_BRRC(
                    sn_list=sn_list, instance=instance, inference_correct=inference_correct,
                    num_fired_output=len(output_neuron_fire_info[instance]["neuron_idx"]),
                    supervised_hidden=supervised_hidden, supervised_output=supervised_output,
                    f_handle=f_handle, 
                    PreSynapticIdx_intended=PreSynapticIdx_intended,
                    PreSynapticIdx_nonintended=PreSynapticIdx_nonintended,
                    desired_ff_idx=desired_ff_neuron[instance]["ff_neuron"],
                    min_fire_time = min_fire_time, 
                    f2f_neuron_lst=f2f_neuron_lst,
                    non_f2f_neuron_lst=non_f2f_neuron_lst, 
                    f2f_neuron_idx=f2f_neuron_idx,
                    WeightRAM=WeightRAM, 
                    stop_num=stop_num, coarse_fine_ratio=coarse_fine_ratio,
                    correct_cnt=correct_cnt
    )
        

    if correct_cnt > max_correct_cnt:
        max_correct_cnt = correct_cnt

    f_handle.write("Succesive correct count: {}\n".format(correct_cnt))
    f_handle.write("-------------------------------------------------\n")

    if correct_cnt == stop_num and (supervised_hidden or supervised_output):
        print("Supervised Training stops at Instance {} because successive correct count has reached {}"
                .format(instance, correct_cnt))
        break
f_handle.write("Maximum successive correct count:{}\n".format(max_correct_cnt))
print("inference_correct list = \n{}\n".format(inference_correct))
print("Supervised Training stops at Instance {}"
        .format(instance))


#%% Dump initial weight vector and final weightRAM
f_handle.write("***************************Weight Change*********************\n")
f_handle.write("Synapse\t\t InitialWeight\t\t FinalWeight\t\t\n")

for synapse_addr in WeightRAM.synapse_addr:
    f_handle.write("{}\t\t {}\t\t\t {}\t\t\n"
                .format(synapse_addr, weight_vector[synapse_addr], WeightRAM.weight[synapse_addr]))
f_handle.write("************************************************************\n")
f_handle.close()
print("Maximum successive correct count:{}\n".format(max_correct_cnt))
print("End of Program!")
            

#%% 
if plot_response:
    plotNeuronResponse_iterative(sn_list=sn_list, neuron_list=[20,21,22,23], instance_list = [0,instance])
if plot_InLatency:
    plotInLatencyDistribution(early_latency_list, late_latency_list, tau_u, num_bins=8)
if plot_response or plot_InLatency:
    plt.show()