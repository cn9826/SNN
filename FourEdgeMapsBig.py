import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import NetworkConnectivity

#%%

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
    
def getInLatencies(in_pattern, early_latency_list, late_latency_list,
                    mean_early, std_early, mean_late, std_late, low_lim=0, high_lim=64,
                    num_edge_maps=4, num_blocks=16):

    # num_in_neurons = num_edge_maps * num_blocks
    in_pattern_list = ["9", "8", "6", "3"]
    num_in_neurons = num_edge_maps * num_blocks
    if not in_pattern in in_pattern_list:
        print("Error when calling getInLatencies: illegal specification of \"in_pattern\"")
        exit(1)
    
    # InLatencies[edge_map_idx][block_idx]
    InLatencies = [[None]*num_blocks] * num_edge_maps
    for map_idx in range(num_edge_maps):
        for block_idx in range(num_blocks):
            latency_late = \
                BimodalLatency("late", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[map_idx][block_idx] = latency_late
            late_latency_list.append(latency_late)
    
    if in_pattern == "9":
        # feature map 0: '--'
        for block_idx in [1, 2]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [1, 10, 13]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in [5, 6, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [2, 5, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 
            
    elif in_pattern == "6":
        # feature map 0: '--'
        for block_idx in [13, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [2, 5, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in [5, 9, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [5, 10, 13]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 

    elif in_pattern == "8":
        # feature map 0: '--'
        for block_idx in [1, 2, 13, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [6, 9]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in []:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [5, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 

    elif in_pattern == "3":
        # feature map 0: '--'
        for block_idx in [1, 2, 13, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [6, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in []:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [2, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 
    

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
printout_dir = "sim_printouts/FourEdgeMapsBig/"
sheet_dir = "sim_printouts/FourEdgeMapsBig/ConnectivityTable.xlsx"

num_categories = 4
num_edge_maps = 4
W_input = 4
F_hidden = 2
S_hidden = 1
depth_hidden_per_sublocation = 4

## Specify common Spiking Neuron Parameters
duration = 80
tau_u = 8      # in units with respect to duration
tau_v = None     # in units with respect to duration
vth_input = 1
vth_hidden = 40 + 16     # with 2-spike consideration: [(2-1) x 5 x tau_u, 2 x 5 x tau_u)
                         # with 2-spike consideration: [(2-1) x 7 x tau_u, 2 x 7 x tau_u)

vth_output = 70          # with 4-spike consideration: [(4-1) x 5 x tau_u, 4 x 5 x tau_u)  
                         # with 4-spike consideration: [(4-1) x 7 x tau_u, 4 x 7 x tau_u)  
## Supervised Training Parameters
supervised_hidden = 1      # turn on/off supervised training in hidden layer
supervised_output = 1      # turn on/off supervised training in output layer 
separation_window = 10
stop_num = 100
coarse_fine_ratio=0.05

## Training Dataset Parameters
num_instances = 10             # number of training instances per epoch

## Simulation Settings
debug_mode = 1
plot_InLatency = 0

if supervised_hidden or supervised_output:
    printout_dir = printout_dir + "Supervised/dumpsim.txt"
else:
    printout_dir = printout_dir + "Inference/dumpsim.txt"

if debug_mode:
    f_handle = open(printout_dir, "w+")
    f_handle.write("supervised_hidden: {}\n".format(supervised_hidden))
    f_handle.write("supervised_output: {}\n".format(supervised_output))
else:
    f_handle = None


## Initialize Connectivity
input_connectivity, hidden_connectivity, output_connectivity, writer = \
    NetworkConnectivity.initializeNetWorkConnectivity(
        num_categories=num_categories, num_edge_maps=num_edge_maps, W_input=W_input,
        F_hidden=F_hidden, S_hidden=S_hidden, 
        depth_hidden_per_sublocation=depth_hidden_per_sublocation,
        sheet_dir=sheet_dir
    )

num_input_neurons = len(input_connectivity)
num_hidden_neurons = len(hidden_connectivity)
num_output_neurons = len(output_connectivity)
num_neurons = num_input_neurons + num_hidden_neurons + num_output_neurons

inital_weight_input = [10] * num_input_neurons 
initial_weight_hidden = [5] * num_hidden_neurons * F_hidden**2 * num_edge_maps
initial_weight_output = [5] * num_output_neurons * num_hidden_neurons 
weight_vector = \
    [
        *inital_weight_input, *initial_weight_hidden, *initial_weight_output
    ]
######################################################################################

#%% Generate Input & Output patterns 
######################################################################################
## Define Input & Output Patterns
mean_early = 0*2*tau_u + 2*tau_u
std_early = int(2*tau_u/3)
mean_late = 4*2*tau_u - 2*tau_u
std_late = int(2*tau_u/3)

input_patterns = ("3", "6", "8", "9")
output_pattern = \
    {
        "3"     :   num_input_neurons + num_hidden_neurons,
        "6"     :   num_input_neurons + num_hidden_neurons + 1,
        "8"     :   num_input_neurons + num_hidden_neurons + 2,
        "9"     :   num_input_neurons + num_hidden_neurons + 3
    }

## Create stimulus spikes at the inuput layer (layer 0)
# Dimension: num_instances x num_input_neurons
stimulus_time_vector = [
                            {
                                "in_pattern"    :    None,
                                "in_latency"    :    [None] * num_input_neurons
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
                           early_latency_list=early_latency_list,
                           late_latency_list=late_latency_list,
                           mean_early=mean_early, std_early=std_early,
                           mean_late=mean_late, std_late=std_late,
                           low_lim=0, high_lim=mean_late+2*tau_u,
                           num_edge_maps=num_edge_maps, num_blocks=W_input**2  
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
    if len(stimulus_time_vector[0]["in_latency"]) != num_input_neurons:
        print("Error: Dimension of stimulus does not match the number of input neurons")
        exit(1)

if len(desired_ff_neuron) != num_instances:
    print("Error: Dimension of desired output time vector does not match the number of instances per epoch")
    exit(1)

######################################################################################

#%% Instantiate ConnectivityTable, WeightRAM and PotentialRAM
######################################################################################
## Initialize Connectivity Table -- consolidate input, hidden and output connectivity 
ConnectivityTable = SNN.ConnectivityInfo(num_neurons=num_neurons)

for i in range(num_input_neurons):
    neuron_idx = i
    ConnectivityTable.layer_num[neuron_idx] = 0
    ConnectivityTable.fan_in_neuron_idx[neuron_idx] = input_connectivity[i]["fan_in_neuron_indices"]
    ConnectivityTable.fan_in_synapse_addr[neuron_idx] = input_connectivity[i]["fan_in_synapse_addrs"]
    ConnectivityTable.fan_out_neuron_idx[neuron_idx] = input_connectivity[i]["fan_out_neuron_indices"]
    ConnectivityTable.fan_out_synapse_addr[neuron_idx] = input_connectivity[i]["fan_out_synapse_indices"]

for i in range(num_hidden_neurons):
    neuron_idx = num_input_neurons + i
    ConnectivityTable.layer_num[neuron_idx] = 1
    ConnectivityTable.fan_in_neuron_idx[neuron_idx] = hidden_connectivity[i]["fan_in_neuron_indices"]
    ConnectivityTable.fan_in_synapse_addr[neuron_idx] = hidden_connectivity[i]["fan_in_synapse_addrs"]
    ConnectivityTable.fan_out_neuron_idx[neuron_idx] = hidden_connectivity[i]["fan_out_neuron_indices"]
    ConnectivityTable.fan_out_synapse_addr[neuron_idx] = hidden_connectivity[i]["fan_out_synapse_indices"]

for i in range(num_output_neurons):
    neuron_idx = num_input_neurons + num_hidden_neurons + i
    ConnectivityTable.layer_num[neuron_idx] = 2
    ConnectivityTable.fan_in_neuron_idx[neuron_idx] = output_connectivity[i]["fan_in_neuron_indices"]
    ConnectivityTable.fan_in_synapse_addr[neuron_idx] = output_connectivity[i]["fan_in_synapse_addrs"]
    ConnectivityTable.fan_out_neuron_idx[neuron_idx] = output_connectivity[i]["fan_out_neuron_indices"]
    ConnectivityTable.fan_out_synapse_addr[neuron_idx] = output_connectivity[i]["fan_out_synapse_indices"]


## Initialize WeightRAM
num_synapses = output_connectivity[-1]["fan_in_synapse_addrs"][-1] + 1 
WeightRAM = SNN.WeightRAM(num_synapses=num_synapses)
for synapse_addr in range(num_synapses):
    post_neuron_idx_WRAM, _ = index_2d(ConnectivityTable.fan_in_synapse_addr, synapse_addr)
    WeightRAM.post_neuron_idx[synapse_addr] = post_neuron_idx_WRAM
    WeightRAM.weight[synapse_addr] = weight_vector[synapse_addr]
    if synapse_addr >= num_input_neurons:
        pre_neuron_idx_WRAM, _ = index_2d(ConnectivityTable.fan_out_synapse_addr, synapse_addr)
        WeightRAM.pre_neuron_idx[synapse_addr] = pre_neuron_idx_WRAM

## Initialize PotentialRAM
PotentialRAM = SNN.PotentialRAM(num_neurons=num_neurons)
PotentialRAM.fan_out_synapse_addr = ConnectivityTable.fan_out_synapse_addr
######################################################################################


#%% Instantiate a list of SpikingNeuron objects
######################################################################################
sn_list = [None] * num_neurons
for neuron_idx in range(num_neurons):
    layer_idx = ConnectivityTable.layer_num[neuron_idx]
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
                                training_on=0,
                                supervised=0
                                )
    elif layer_idx == 1:
        depth_causal = 2
        depth_anticausal = 14
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
                                training_on=training_on,
                                supervised=supervised
                                )

    elif layer_idx == 2:
        depth_causal = 2
        depth_anticausal = 6
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
                                training_on=training_on,
                                supervised=supervised
                                )
    sn_list[neuron_idx] = sn
######################################################################################

#%% Inter-neuron data initialization 
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
            "causal"        :   [],
            "anti-causal"   :   []
        }         
        for i in range(num_instances)
    ]

PreSynapticIdx_nonintended = \
    [
        {
            "causal"        :   [],
            "anti-causal"   :   []
        } for i in range(num_instances)
    ]
    
## Loop instances
for instance in range(num_instances):
    if debug_mode:
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
        
        for i in range(num_neurons):
            sn_list[i].accumulate(sim_point=sim_point, 
                                    spike_in_info=spike_info[instance][sim_point], 
                                    WeightRAM_inst=WeightRAM,
                                    debug_mode=debug_mode,
                                    instance=instance,
                                    f_handle=f_handle
                                    )                                            
            # upadate the current potential to PotentialRAM
            PotentialRAM.potential[i] = sn_list[i].v[sim_point]
            
            # update the list of synapses that fired at this sim_point
            if (sn_list[i].fire_cnt != -1):
                if (sn_list[i].spike_out_info[sn_list[i].fire_cnt]["time"] == sim_point):
                    if (len(sn_list[i].fan_out_synapse_addr) == 1): # if single fan-out
                        fired_synapse_list[instance][sim_point].append(val)
                        spike_info[instance][sim_point]["fired_synapse_addr"].append(val)
                    else:   # if multiple fan-out synpases
                        for key,val in enumerate(sn_list[i].fan_out_synapse_addr):
                            fired_synapse_list[instance][sim_point].append(val)
                            spike_info[instance][sim_point]["fired_synapse_addr"].append(val)
                    
                    fired_neuron_list[instance][sim_point].append(sn_list[i].neuron_idx)

                    # if the fired neuron at this sim_point is in the hidden layer
                    if (sn_list[i].layer_idx == 1):
                        hidden_neuron_fire_info[instance]["neuron_idx"].append(sn_list[i].neuron_idx)
                        hidden_neuron_fire_info[instance]["time"].append(sim_point)

                    # if the fired neuron at this sim_point is in the output layer
                    if (sn_list[i].layer_idx == 2):
                        output_neuron_fire_info[instance]["neuron_idx"].append(sn_list[i].neuron_idx)
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
    if debug_mode:
        f_handle.write("Succesive correct count: {}\n".format(correct_cnt))
        f_handle.write("-------------------------------------------------\n")

    if correct_cnt == stop_num and (supervised_hidden or supervised_output):
        print("Supervised Training stops at Instance {} because successive correct count has reached {}"
                .format(instance, correct_cnt))
        break
    
    ## clear the state varaibles of sn_list and PotentialRAM
    PotentialRAM.clearPotential()
    for i in range(num_neurons):
        sn_list[i].clearStateVariables()

if debug_mode:
    f_handle.write("Maximum successive correct count:{}\n".format(max_correct_cnt))

print("inference_correct list = \n{}\n".format(inference_correct))
print("Supervised Training stops at Instance {}"
        .format(instance))


#%% Dump initial weight vector and final weightRAM
if debug_mode:
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
if plot_InLatency:
    plotInLatencyDistribution(early_latency_list, late_latency_list, tau_u, num_bins=8)
    plt.show()
