import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter
from pandas import DataFrame

#%% Defined Functions 
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



#%% Initializations of Tables and Objects and Lists of Objects
#####################################################################################################################
printout_dir = "sim_printouts/XOR/"

## Specify Global Connectivity Parmeters
num_neurons_perLayer = [2,4,2]                           # Assuming num_neurons_perLayer is the number of connections in FC case
max_num_fires = 1

## Specify common Spiking Neuron Parameters
duration = 200
tau_u = 16      # in units with respect to duration
tau_v = None     # in units with respect to duration
threshold = 120

## Supervised Training Parameters
supervised_training_on = 1      # turn on/off supervised training 
separation_window = tau_u
stop_num = 20
coarse_fine_ratio=0.5

## Training Dataset Parameters
num_epochs = 1                  # number of epochs
num_instances =100              # number of training instances per epoch

## Simulation Settings
debug_mode = 1
plot_response = 1


## Define Input & Output Patterns
input_pattern = \
    {
        "00"    :   [0, 0],
        "01"    :   [0, 30],        # 3*tau_u
        "10"    :   [30, 0],
        "11"    :   [30, 30]
    }

## Define Output first-to-fire pattern
output_pattern = \
    {
        "0"     :   sum(num_neurons_perLayer[0:-1]),
        "1"     :   sum(num_neurons_perLayer[0:-1]) + 1
    }

input_output_map = \
    {
        "00"    :   "0",
        "01"    :   "1",
        "10"    :   "1",
        "11"    :   "0"
    }
## Create stimulus spikes at the inuput layer (layer 0)
# Dimension: num_epochs x num_instances x num_input_neurons
stimulus_time_vector = [
                                [
                                    {
                                        "in_pattern"       :    None,
                                        "in_latency"    :    [None] * num_neurons_perLayer[0]
                                    }
                                    for instance in range(num_instances)
                                ]  for epoch in range(num_epochs)
                       ]
for epoch in range (num_epochs):
    for instance in range(num_instances):
        stimulus_time_vector[epoch][instance]["in_pattern"] = \
                random.choice(list(input_pattern.keys()))
        stimulus_time_vector[epoch][instance]["in_latency"] = \
input_pattern[stimulus_time_vector[epoch][instance]["in_pattern"]]


## Specify the index of the desired output layer neuron to fire
desired_ff_neuron = [
                        [
                            {
                                "in_pattern"            :   None,
                                "out_pattern"           :   None,
                                "ff_neuron"             :   None
                            } for instance in range(num_instances)
                        ] for epoch in range(num_epochs)
                    ]

for epoch in range(num_epochs):
    for instance in range(num_instances):
        desired_ff_neuron[epoch][instance]["in_pattern"] = \
            stimulus_time_vector[epoch][instance]["in_pattern"]
        desired_ff_neuron[epoch][instance]["out_pattern"] = \
            input_output_map.get(stimulus_time_vector[epoch][instance]["in_pattern"])
        desired_ff_neuron[epoch][instance]["ff_neuron"] = \
            output_pattern[desired_ff_neuron[epoch][instance]["out_pattern"]]



#####################################################################################################################
if supervised_training_on:
    printout_dir = printout_dir + "Supervised/BRRC/dumpsim.txt"
else:
    printout_dir = printout_dir + "Inference/BRRC/dumpsim.txt"
f_handle = open(printout_dir, "w+")


if len(stimulus_time_vector) != num_epochs:
    print("Error: Dimension of stimulus does not match the number of epochs!")
    exit(1)
else:
    if len(stimulus_time_vector[0]) != num_instances:
        print("Error: Dimension of stimulus does not match the number of instances per epoch")
        exit(1)
    else:
        if len(stimulus_time_vector[0][0]["in_latency"]) != num_neurons_perLayer[0]:
            print("Error: Dimension of stimulus does not match the number of input neurons")
            exit(1)

if len(desired_ff_neuron) != num_epochs:
    print("Error: Dimension of desired output time vector does not match the number of epochs!")
    exit(1)
else:
    if len(desired_ff_neuron[0]) != num_instances:
        print("Error: Dimension of desired output time vector does not match the number of instances per epoch")
        exit(1)

stimulus_vector_info =  [
                            [                            
                                [] 
                                for instance in range(num_instances)
                            ] for epoch in range(num_epochs)
                        ] 
for epoch in range(num_epochs):
    for instance in range(num_instances):
        for synapse_idx in range(num_neurons_perLayer[0]):
            stimulus_entry = {}
            stimulus_entry["fan_in_synapse_addr"] = synapse_idx
            stimulus_entry["time"] = stimulus_time_vector[epoch][instance]["in_latency"][synapse_idx]
            stimulus_vector_info[epoch][instance].append(stimulus_entry)

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

## Initialize WeightRAM -- a class with list attributes sorted/indexed by fan_in_synapse_addr; its contents are weight & neuron_idx
# weight_vector = [
#                     10, 10,
#                     8, 8, 10, 0, 0, 10, 8, 8,
#                     10, 10, 10, 10, 10, 10, 10, 10
#                 ]     # used for unsupervised STDP initialization

# randomIntList=randomInt(8,3,8)
# weight_vector = [
#                     10, 10,
#                     *randomIntList,
#                     10, 10, 10, 10, 10, 10, 10, 10
#                 ]     # used for unsupervised STDP initialization

weight_vector = [
                    10, 10,
                    15, 15, 10, -15, -15, 10, 15, 15,
                    6, 6, 6, 6, 6, 6, 6, 6
                ] 

# weight_vector = [
#                     10, 10,
#                     15, 15, 10, -15, -15, 10, 15, 15,
#                     6, 0, 0, 6, 5, 8, 10, 5
#                 ]     #  one set that works under threshold 120, input pattern [0, 40]
                        ## BR_training: A_coarse_comp=5, A_fine_comp=2, tau_long=10, tau_short=4, A_coarse=4, A_fine=1, tau=14

# weight_vector = [
#                     10, 10,
#                     15, 15, 10, -15, -15, 10, 15, 15,
#                     6, -5, -5, 6, 3, 15, 15, 3
#                 ]     #  one set that works under threshold 120, input pattern [0, 40]
                        ## BRRC_training: A_coarse_comp=5, A_fine_comp=2, tau_long=10, tau_short=4, A_coarse=4, A_fine=1, tau=14


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
## and specify the desired out-spike time for output-layer neurons
# row represents each training instance (epoch)
# col represents each Spiking Neuron object sorted by ascending index 
sn_list =   [
                [
                    []
                    for instance in range(num_instances)
                ] for epochs in range(num_epochs)
            ]
for epoch in range(num_epochs):
    for instance in range(num_instances):
        for i in neuron_indices:
            sn = SNN.SpikingNeuron( layer_idx=ConnectivityTable.layer_num[i],
                                    neuron_idx=i, 
                                    fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[i],
                                    fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[i],
                                    tau_u=tau_u,
                                    tau_v=tau_v,
                                    threshold=threshold,
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
                sn.training_on = supervised_training_on
                sn.supervised = 1
                sn.desired_ff_idx =         [
                                                [
                                                    desired_ff_neuron[epoch][instance]["ff_neuron"]
                                                    for instance in range(num_instances)
                                                ] for epoch in range(num_epochs)
                                            ]
               
            sn_list[epoch][instance].append(sn)
#%% Simulation of SNN in time-major order

## Processing spikes in the network
fired_synapse_list =    [
                            [   
                                [
                                    [] for step in range(0, sn.duration, sn.dt)
                                ]
                                for instance in range(num_instances)
                            ] for epoch in range(num_epochs)
                        ]   # num_epochs x num_instances x num_timesteps x num_fired_synapses

fired_neuron_list =    [
                            [   
                                [
                                    [] for step in range(0, sn.duration, sn.dt)
                                ]
                                for instance in range(num_instances)
                            ] for epoch in range(num_epochs)
                        ]   # num_epochs x num_instances x num_timesteps x num_fired_neurons

spike_info = [
                [   
                    [
                        {    
                            "fired_synapse_addr": [],
                            "time"              : None
                        }
                        for step in range(0, sn.duration, sn.dt)
                    ] for instance in range(num_instances)
                ] for epoch in range(num_epochs)
             ]            
             # a list of dictionaries sorted by time steps

## Initialize statistics
output_neuron_fire_info =   [
                                [ 
                                    {
                                        "neuron_idx":   [],
                                        "time"      :   []
                                    } 
                                    for instance in range(num_instances)
                                ]
                                for epoch in range(num_epochs)
                            ]   

inference_correct =     [
                            [    
                                None for instance in range(num_instances) 
                            ] for epoch in range(num_epochs)
                        ]   # num_epochs x num_instances

## Training Loop
for epoch in range(num_epochs):
    f_handle.write("********************************Beginning of Epoch {}!***********************\n".format(epoch))
    correct_cnt = 0
    for instance in range(num_instances):
        f_handle.write("---------------Instance {} {} -----------------\n".format(instance,stimulus_time_vector[epoch][instance]["in_pattern"]))
        for sim_point in range(0, sn.duration, sn.dt):
            # first check if any input synpase fires at this time step
            if sim_point in stimulus_time_vector[epoch][instance]["in_latency"]:
                fired_synapse_list[epoch][instance][sim_point].extend(
                    index_duplicate(stimulus_time_vector[epoch][instance]["in_latency"], sim_point)
                )
                spike_info[epoch][instance][sim_point]["fired_synapse_addr"].extend(
                    index_duplicate(stimulus_time_vector[epoch][instance]["in_latency"], sim_point)
                )

            spike_info[epoch][instance][sim_point]["time"] = sim_point
            
            for i in neuron_indices:
                sn_list[epoch][instance][i].accumulate(sim_point=sim_point, 
                                        spike_in_info=spike_info[epoch][instance][sim_point], 
                                        WeightRAM_inst=WeightRAM,
                                        debug_mode=debug_mode,
                                        instance=instance,
                                        f_handle=f_handle
                                        )                               
                
                # upadate the current potential to PotentialRAM
                PotentialRAM.potential[instance][i] = sn_list[epoch][instance][i].v[sim_point]
                
                # update the list of synapses that fired at this sim_point
                if (sn_list[epoch][instance][i].fire_cnt != -1):
                    if (sn_list[epoch][instance][i].spike_out_info[sn_list[epoch][instance][i].fire_cnt]["time"] == sim_point):
                        if (len(sn_list[epoch][instance][i].fan_out_synapse_addr) == 1): # if single fan-out
                            fired_synapse_list[epoch][instance][sim_point].append(val)
                            spike_info[epoch][instance][sim_point]["fired_synapse_addr"].append(val)
                        else:   # if multiple fan-out synpases
                            for key,val in enumerate(sn_list[epoch][instance][i].fan_out_synapse_addr):
                                fired_synapse_list[epoch][instance][sim_point].append(val)
                                spike_info[epoch][instance][sim_point]["fired_synapse_addr"].append(val)
                        
                        fired_neuron_list[epoch][instance][sim_point].append(sn_list[epoch][instance][i].neuron_idx)

                        # if the fired neuron at this sim_point is in the output layer
                        if (sn_list[epoch][instance][i].layer_idx == num_layers - 1):
                            output_neuron_fire_info[epoch][instance]["neuron_idx"].append(sn_list[epoch][instance][i].neuron_idx)
                            output_neuron_fire_info[epoch][instance]["time"].append(sim_point)

        # at the end of the simulation duration, inspect output-layer firing info         
        if len(output_neuron_fire_info[epoch][instance]["neuron_idx"]) > 0:
            # find the minimum of firing time and its corresponding list index
            min_fire_time = min(output_neuron_fire_info[epoch][instance]["time"])
            list_min_idx = index_duplicate(output_neuron_fire_info[epoch][instance]["time"], min_fire_time)
            f2f_neuron_idx =    [
                                    output_neuron_fire_info[epoch][instance]["neuron_idx"][list_idx]
                                    for list_idx in list_min_idx
                                ] 

            # find the non-F2F neurons that fired within separation window
            non_f2f_neuron_idx = [  
                                    neuron_idx
                                    for list_idx, neuron_idx in enumerate(output_neuron_fire_info[epoch][instance]["neuron_idx"])
                                    if output_neuron_fire_info[epoch][instance]["time"][list_idx] - min_fire_time > 0
                                    and output_neuron_fire_info[epoch][instance]["time"][list_idx] - min_fire_time <= separation_window
                                 ]

            # more than one output layer neuron have fired at the same min_fire_time
            if len(list_min_idx) > 1:       
                correct_cnt = 0
                inference_correct[epoch][instance] = 0
                
                if supervised_training_on == 1:
                    # iterate through all the f2f neuron index
                    for neuron_idx in f2f_neuron_idx:
                        isf2f = 1
                        sn = sn_list[epoch][instance][neuron_idx]
                        
                        # if the f2f neuron is the intended, F2F P+ quadrant
                        if neuron_idx == desired_ff_neuron[epoch][instance]["ff_neuron"]:
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
            
            # only one output layer neuron fired at the min_fire_time             
            elif len(list_min_idx) == 1:
                neuron_idx = f2f_neuron_idx[0]
                sn = sn_list[epoch][instance][neuron_idx] 
                
                # if the F2F neuron is the intended 
                if neuron_idx == desired_ff_neuron[epoch][instance]["ff_neuron"]:
                    correct_cnt += 1
                    inference_correct[epoch][instance] = 1
                
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
                                sn_nonf2f = sn_list[epoch][instance][neuron_idx] 
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
                    inference_correct[epoch][instance] = 0
                    
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
                        if desired_ff_neuron[epoch][instance]["ff_neuron"] in non_f2f_neuron_idx:
                            isf2f = 0
                            reward_signal = 0
                            isIntended = 1
                            neuron_idx = desired_ff_neuron[epoch][instance]["ff_neuron"]
                            sn_nonf2f = sn_list[epoch][instance][neuron_idx]
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
                
        else:
        # no output layer neuron has fired
            correct_cnt=0
            inference_correct[epoch][instance] = 0
            f_handle.write("Instance {}: training failed -- no output layer neuron has fired up until sim_point{}!\n"
            .format(instance, sim_point))
            print("Instance {}: training failed -- no output layer neuron has fired up until sim_point{}!"
            .format(instance, sim_point))
        
        f_handle.write("Succesive correct count: {}\n".format(correct_cnt))
        f_handle.write("-------------------------------------------------\n")

        if correct_cnt == stop_num and supervised_training_on:
            print("Supervised Training stops at Epoch {} Instance {} because successive correct count has reached {}"
                    .format(epoch, instance, correct_cnt))
            break
    f_handle.write("********************************End of Epoch {}!***********************\n\n".format(epoch))

f_handle.write("{}\n\n".format(inference_correct))
print("{}\n".format(inference_correct))
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
    plotNeuronResponse_iterative(sn_list=sn_list, epochs_list=[num_epochs-1], instance_list = [instance-1], only_output_layer=0)
    plt.show()
print("End of Program!")
