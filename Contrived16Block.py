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
    
def getInLatencies(in_pattern, num_in_neurons, 
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

    if in_pattern == "O":
        for i in [0, 3, 5, 6]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
    elif in_pattern == "X":
        for i in [1, 2, 4, 7]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
    elif in_pattern == "<<":
        for i in [0, 1, 6, 7]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
    elif in_pattern == "//":
        for i in [0, 1, 2, 3]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)    
    elif in_pattern == ">>":
        for i in [2, 3, 4, 5]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)    
    elif in_pattern == "UA":
        for i in [0, 2, 5, 7]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)    
    elif in_pattern == "DA":
        for i in [1, 3, 4, 6]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)    
    elif in_pattern == "BS":
        for i in [4, 5, 6, 7]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)    
    elif in_pattern == "Bad//":
        for i in [0, 1, 2, 6]:
            InLatencies[i] = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)    
    return InLatencies

#%% Parameters to tune
######################################################################################
printout_dir = "sim_printouts/Contrived16Block/"

## Specify Global Connectivity Parmeters
num_neurons_perLayer = [8,6]                           # Assuming num_neurons_perLayer is the number of connections in FC case
max_num_fires = 1

## Specify common Spiking Neuron Parameters
duration = 80
tau_u = 8      # in units with respect to duration
tau_v = None     # in units with respect to duration
vth_low = 1
vth_high = 140

## Supervised Training Parameters
supervised_training_on = 1      # turn on/off supervised training 
separation_window = 12
stop_num = 100
coarse_fine_ratio=0.8

## Training Dataset Parameters
num_instances =2000              # number of training instances per epoch

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
mean_early = 0*2*tau_u + 2*tau_u
std_early = int(4*tau_u/3)
mean_late = 4*2*tau_u - 2*tau_u
std_late = int(4*tau_u/3)

initial_weight = [6] * num_neurons_perLayer[-2] * num_neurons_perLayer[-1] 
weight_vector = \
    [
        10, 10, 10, 10, 10, 10, 10, 10,
        *initial_weight
    ]

input_patterns = ("O", "X", "<<", "//", ">>", "Bad//")

output_pattern = \
    {
        "O"      :   sum(num_neurons_perLayer[0:-1]),
        "X"      :   sum(num_neurons_perLayer[0:-1]) + 1,
        "<<"     :   sum(num_neurons_perLayer[0:-1]) + 2,
        "//"     :   sum(num_neurons_perLayer[0:-1]) + 3,
        ">>"     :   sum(num_neurons_perLayer[0:-1]) + 4,
        "Bad//"  :   sum(num_neurons_perLayer[0:-1]) + 5
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
            random.choice(input_patterns)
    stimulus_time_vector[instance]["in_latency"] = \
            getInLatencies(in_pattern=stimulus_time_vector[instance]["in_pattern"],
                           num_in_neurons=num_neurons_perLayer[0],
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

WeightRAM = SNN.WeightRAM(num_synapses)
for i in synapse_addr:
    neuron_idx_WRAM, connection_num = index_2d(ConnectivityTable.fan_in_synapse_addr,synapse_addr[i])
    num_fan_in = len([fan_in for fan_in in ConnectivityTable.fan_in_synapse_addr[neuron_idx_WRAM] if fan_in is not None]) 
    # if hidden layer, de-uniformize fan-in weights
    WeightRAM.neuron_idx[i] = neuron_idx_WRAM
    WeightRAM.weight[i] = weight_vector[i]

## Initialize PotentialRAM --  a class with list attibutes sorted/indexed by neuron_idx; its contents are membrane potential & fan_out_synapse_addr
PotentialRAM = SNN.PotentialRAM(
                                num_neurons=num_neurons, 
                                max_num_connections=max(num_neurons_perLayer),
                                num_instances=num_instances,
                               )
PotentialRAM.fan_out_synapse_addr = ConnectivityTable.fan_out_synapse_addr

## initialize a list of all SpikingNeuron objects
sn_list =   [
                []
                for instance in range(num_instances)
            ]
for instance in range(num_instances):
    for i in neuron_indices:
        sn = SNN.SpikingNeuron( layer_idx=ConnectivityTable.layer_num[i],
                                neuron_idx=i, 
                                fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[i],
                                fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[i],
                                tau_u=tau_u,
                                tau_v=tau_v,
                                threshold=vth_low,
                                duration=duration,
                                max_num_fires=max_num_fires,
                                training_on=0,
                                supervised=0
                                )
        # chekc if neuron is in the input layer
        if sn.layer_idx == 0:
            sn.training_on = 0

        # check if neuron is in the output layer 
        if sn.layer_idx == num_layers - 1:
            sn.threshold = vth_high
            sn.training_on = supervised_training_on
            sn.supervised = 1
            
        sn_list[instance].append(sn)
######################################################################################
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
for instance in range(num_instances):
    f_handle.write("---------------Instance {} {} -----------------\n".format(instance,stimulus_time_vector[instance]["in_pattern"]))
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

                    # if the fired neuron at this sim_point is in the output layer
                    if (sn_list[instance][i].layer_idx == num_layers - 1):
                        output_neuron_fire_info[instance]["neuron_idx"].append(sn_list[instance][i].neuron_idx)
                        output_neuron_fire_info[instance]["time"].append(sim_point)

    # at the end of the simulation duration, inspect output-layer firing info         
    if len(output_neuron_fire_info[instance]["neuron_idx"]) > 0:
        # find the minimum of firing time and its corresponding list index
        min_fire_time = min(output_neuron_fire_info[instance]["time"])
        list_min_idx = index_duplicate(output_neuron_fire_info[instance]["time"], min_fire_time)
        f2f_neuron_idx =    [
                                output_neuron_fire_info[instance]["neuron_idx"][list_idx]
                                for list_idx in list_min_idx
                            ] 

        # find the non-F2F neurons that fired within separation window
        non_f2f_neuron_idx = [  
                                neuron_idx
                                for list_idx, neuron_idx in enumerate(output_neuron_fire_info[instance]["neuron_idx"])
                                if output_neuron_fire_info[instance]["time"][list_idx] - min_fire_time > 0
                                and output_neuron_fire_info[instance]["time"][list_idx] - min_fire_time <= separation_window
                                ]

        # more than one output layer neuron have fired at the same min_fire_time
        if len(list_min_idx) > 1:       
            correct_cnt = 0
            inference_correct[instance] = 0
            
            if supervised_training_on == 1:
                # iterate through all the f2f neuron index
                for neuron_idx in f2f_neuron_idx:
                    isf2f = 1
                    sn = sn_list[instance][neuron_idx]
                    
                    # if the f2f neuron is the intended, F2F P+ quadrant
                    if neuron_idx == desired_ff_neuron[instance]["ff_neuron"]:
                        reward_signal = 1
                        isIntended = 1
                        newWeight = sn.BRRC_training(
                                                    spike_ref_time=min_fire_time,
                                                    spike_out_time=sn.spike_out_info[0]["time"],                        
                                                    instance=instance,
                                                    oldWeight=sn.oldWeight,
                                                    causal_fan_in_addr=sn.causal_fan_in_addr,
                                                    f_handle=f_handle,
                                                    reward_signal=reward_signal, 
                                                    isf2f=isf2f, isIntended=isIntended,
                                                    successive_correct_cnt=correct_cnt,
                                                    coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                    debug=1
                                                    )
                        sn.updateWeight(fan_in_addr=sn.causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight)

                    # if the f2f neuron is not the intended, F2F P- quadrant                                
                    else:
                        reward_signal = 0
                        isIntended = 0
                        newWeight = sn.BRRC_training(
                                                    spike_ref_time=sn.causal_spike_in_info["time"],
                                                    spike_out_time=sn.spike_out_info[0]["time"],                        
                                                    instance=instance,
                                                    oldWeight=sn.oldWeight,
                                                    causal_fan_in_addr=sn.causal_fan_in_addr,
                                                    f_handle=f_handle,
                                                    reward_signal=reward_signal, 
                                                    isf2f=isf2f, isIntended=isIntended,
                                                    successive_correct_cnt=correct_cnt,
                                                    coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                    debug=1
                                                    )
                        sn.updateWeight(fan_in_addr=sn.causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                       
                    # apply Non-F2F P- upadate only on the desired neuron 
                    if not desired_ff_neuron[instance]["ff_neuron"] in f2f_neuron_idx:
                        isf2f = 0
                        reward_signal = 0
                        isIntended = 1
                        intended_idx = desired_ff_neuron[instance]["ff_neuron"]
                        sn_intended = sn_list[instance][intended_idx]
                        # prepare information needed for weight update
                        if sn_intended.fire_cnt != -1:
                            sn_intended_out_time = sn_intended.spike_out_info[0]["time"]
                        else:
                            sn_intended_out_time = None
                        (intended_updateAddr, intended_oldWeight, causal_in_time) = \
                            sn_intended.spike_in_cache.getUpdateAddr(isf2f, reward_signal, isIntended)
                        
                        newWeight = sn_intended.BRRC_training(
                                                    spike_ref_time=min_fire_time,
                                                    spike_out_time=sn_intended_out_time,                      
                                                    instance=instance,
                                                    oldWeight=intended_oldWeight,
                                                    causal_fan_in_addr=intended_updateAddr,
                                                    f_handle=f_handle,
                                                    reward_signal=reward_signal, 
                                                    isf2f=isf2f, isIntended=isIntended,
                                                    successive_correct_cnt=correct_cnt,
                                                    coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                    debug=1
                                                    )
                        sn_intended.updateWeight(fan_in_addr=intended_updateAddr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                       
                        
        # only one output layer neuron fired at the min_fire_time             
        elif len(list_min_idx) == 1:
            neuron_idx = f2f_neuron_idx[0]
            sn = sn_list[instance][neuron_idx] 
            
            # if the F2F neuron is the intended 
            if neuron_idx == desired_ff_neuron[instance]["ff_neuron"]:
                correct_cnt += 1
                inference_correct[instance] = 1
            
                if supervised_training_on == 1:    
                    # F2F P+ quadrant 
                    isf2f = 1
                    reward_signal = 1
                    isIntended = 1
                    newWeight = sn.BRRC_training(
                                                spike_ref_time=sn.causal_spike_in_info["time"],
                                                spike_out_time=sn.spike_out_info[0]["time"],                        
                                                instance=instance,
                                                oldWeight=sn.oldWeight,
                                                causal_fan_in_addr=sn.causal_fan_in_addr,
                                                f_handle=f_handle,
                                                reward_signal=reward_signal, 
                                                isf2f=isf2f, isIntended=isIntended,
                                                successive_correct_cnt=correct_cnt,
                                                coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                debug=1
                                                )
                    sn.updateWeight(fan_in_addr=sn.causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                       
                    
                    # then non-F2F P+ quadrant
                    if len(non_f2f_neuron_idx) > 0:
                        isf2f = 0
                        reward_signal = 1
                        isIntended = 0
                        for neuron_idx in non_f2f_neuron_idx:
                            sn_nonf2f = sn_list[instance][neuron_idx] 
                            newWeight = sn_nonf2f.BRRC_training(
                                                        spike_ref_time=min_fire_time,
                                                        spike_out_time=sn_nonf2f.spike_out_info[0]["time"],                        
                                                        instance=instance,
                                                        oldWeight=sn_nonf2f.oldWeight,
                                                        causal_fan_in_addr=sn_nonf2f.causal_fan_in_addr,
                                                        f_handle=f_handle,
                                                        reward_signal=reward_signal, 
                                                        isf2f=isf2f, isIntended=isIntended,
                                                        successive_correct_cnt=correct_cnt,
                                                        coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                        debug=1
                                                        )
                            sn_nonf2f.updateWeight(fan_in_addr=sn_nonf2f.causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                       

            
            # if the F2F neuron is not the intended
            else:
                correct_cnt = 0
                inference_correct[instance] = 0
                
                if supervised_training_on == 1:
                    # F2F P- quadrant 
                    isf2f = 1
                    reward_signal = 0
                    isIntended = 0
                    newWeight = sn.BRRC_training(
                                                spike_ref_time=sn.causal_spike_in_info["time"],
                                                spike_out_time=sn.spike_out_info[0]["time"],                        
                                                instance=instance,
                                                oldWeight=sn.oldWeight,
                                                causal_fan_in_addr=sn.causal_fan_in_addr,
                                                f_handle=f_handle,
                                                reward_signal=reward_signal, 
                                                isf2f=isf2f, isIntended=isIntended,
                                                successive_correct_cnt=correct_cnt,
                                                coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                debug=1
                                                )
                    sn.updateWeight(fan_in_addr=sn.causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                        
                    
                    # then non-F2F P- quadrant, only applied on the intended F2F neuron
                    isf2f = 0
                    reward_signal = 0
                    isIntended = 1
                    intended_idx = desired_ff_neuron[instance]["ff_neuron"]
                    sn_intended = sn_list[instance][intended_idx]
                    # prepare information needed for weight update
                    if sn_intended.fire_cnt != -1:
                        sn_intended_out_time = sn_intended.spike_out_info[0]["time"]
                    else:
                        sn_intended_out_time = None
                    (intended_updateAddr, intended_oldWeight, causal_in_time) = \
                        sn_intended.spike_in_cache.getUpdateAddr(isf2f, reward_signal, isIntended)
                    
                    newWeight = sn_intended.BRRC_training(
                                                spike_ref_time=min_fire_time,
                                                spike_out_time=sn_intended_out_time,                      
                                                instance=instance,
                                                oldWeight=intended_oldWeight,
                                                causal_fan_in_addr=intended_updateAddr,
                                                f_handle=f_handle,
                                                reward_signal=reward_signal, 
                                                isf2f=isf2f, isIntended=isIntended,
                                                successive_correct_cnt=correct_cnt,
                                                coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                                debug=1
                                                )
                    sn_intended.updateWeight(fan_in_addr=intended_updateAddr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                       
            
    else:
    # no output layer neuron has fired
        correct_cnt=0
        inference_correct[instance] = 0
        f_handle.write("Instance {}: no output layer neuron has fired up until sim_point{}!\n"
        .format(instance, sim_point))
        print("Instance {}: no output layer neuron has fired up until sim_point{}!"
        .format(instance, sim_point))

        # non-F2F P- update on the desired 
        isf2f = 0
        reward_signal = 0
        isIntended = 1
        neuron_idx = desired_ff_neuron[instance]["ff_neuron"]
        sn_intended = sn_list[instance][neuron_idx]
        if sn_intended.fire_cnt != -1:
            sn_intended_out_time = sn_intended.spike_out_info[0]["time"]
        else:
            sn_intended_out_time = None                       
        # prepare information needed for weight update
        if sn_intended.fire_cnt != -1:
            sn_intended_out_time = sn_intended.spike_out_info[0]["time"]
        else:
            sn_intended_out_time = None
        (intended_updateAddr, intended_oldWeight, causal_in_time) = \
            sn_intended.spike_in_cache.getUpdateAddr(isf2f, reward_signal, isIntended)
        
        newWeight = sn_intended.BRRC_training(
                                    spike_ref_time=min_fire_time,
                                    spike_out_time=sn_intended_out_time,                      
                                    instance=instance,
                                    oldWeight=intended_oldWeight,
                                    causal_fan_in_addr=intended_updateAddr,
                                    f_handle=f_handle,
                                    reward_signal=reward_signal, 
                                    isf2f=isf2f, isIntended=isIntended,
                                    successive_correct_cnt=correct_cnt,
                                    coarse_fine_cut_off=stop_num*coarse_fine_ratio,
                                    debug=1
                                    )
        sn_intended.updateWeight(fan_in_addr=intended_updateAddr, WeightRAM_inst=WeightRAM, newWeight=newWeight)                       
    
    if correct_cnt > max_correct_cnt:
        max_correct_cnt = correct_cnt

    f_handle.write("Succesive correct count: {}\n".format(correct_cnt))
    f_handle.write("-------------------------------------------------\n")

    if correct_cnt == stop_num and supervised_training_on:
        print("Supervised Training stops at Instance {} because successive correct count has reached {}"
                .format(instance, correct_cnt))
        break
f_handle.write("Maximum successive correct count:{}\n".format(max_correct_cnt))
print("inference_correct list = \n{}\n".format(inference_correct))


#%% Dump initial weight vector and final weightRAM
f_handle.write("***************************Weight Change*********************\n")
f_handle.write("Synapse\t\t InitialWeight\t\t FinalWeight\t\t\n")

for synapse_addr in WeightRAM.synapse_addr:
    f_handle.write("{}\t\t {}\t\t\t {}\t\t\n"
                .format(synapse_addr, weight_vector[synapse_addr], WeightRAM.weight[synapse_addr]))
f_handle.write("************************************************************\n")
f_handle.close()

#%% 
if plot_response:
    plotNeuronResponse_iterative(sn_list=sn_list, neuron_list=[8,9,10,11,12], instance_list = [0, instance])
    plt.show()
print("End of Program!")