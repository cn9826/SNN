import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter
from pandas import DataFrame


f_handle = open("dumpsim.txt", "w+")
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

#%% Initializations of Tables and Objects and Lists of Objects
#####################################################################################################################
## Specify Global Connectivity Parmeters
num_neurons_perLayer = [2,4,2]                           # Assuming num_neurons_perLayer is the number of connections in FC case
max_num_fires = 1

## Specify common Spiking Neuron Parameters
duration = 200
tau_u = 16      # in units with respect to duration
tau_v = None     # in units with respect to duration
threshold = 120

num_epochs = 1                 # number of epochs
num_instances = 1000             # number of training instances per epoch
stop_num = 14
coarse_fine_ratio=0.5
## Define Input & Output Patterns
input_pattern = \
    {
        "00"    :   [0, 0],
        "01"    :   [0, 50],        # 3*tau_u
        "10"    :   [50, 0],
        "11"    :   [50, 50]
    }

output_pattern = \
    {
        "0"     :   [80, 150],     # class 0 neuron fires first 
        "1"     :   [150, 80]      # class 1 neuron fires first
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
        # stimulus_time_vector[epoch][instance]["in_pattern"] = \
        #         "11"
        stimulus_time_vector[epoch][instance]["in_latency"] = \
input_pattern[stimulus_time_vector[epoch][instance]["in_pattern"]]

## Specify desired out-spike times of the last-layer neurons
# Dimension: num_epochs x num_instances x num_output_neurons
desired_out_time_vector =   [
                                [
                                    {
                                        "in_pattern"        :   None,
                                        "out_pattern"       :   None,
                                        "out_latency"       :   [None] * num_neurons_perLayer[-1],
                                        "class_num"         :   None
                                    }                                    
                                    for instance in range(num_instances)
                                ]  for epoch in range(num_epochs)
                            ]
for epoch in range (num_epochs):
    for instance in range(num_instances):
        desired_out_time_vector[epoch][instance]["in_pattern"] = \
            stimulus_time_vector[epoch][instance]["in_pattern"]
        desired_out_time_vector[epoch][instance]["out_pattern"] = \
                input_output_map.get(stimulus_time_vector[epoch][instance]["in_pattern"])
        desired_out_time_vector[epoch][instance]["out_latency"] = \
                output_pattern[desired_out_time_vector[epoch][instance]["out_pattern"]]
        if desired_out_time_vector[epoch][instance]["out_pattern"] == "0":
                desired_out_time_vector[epoch][instance]["class_num"] = 0
        elif desired_out_time_vector[epoch][instance]["out_pattern"] == "1":
                desired_out_time_vector[epoch][instance]["class_num"] = 1

## Initilize synaptic weight for all the synapses
initial_weight = 8

## Simulation Settings
debug_mode = 1
plot_response = 0
#####################################################################################################################
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

if len(desired_out_time_vector) != num_epochs:
    print("Error: Dimension of desired output time vector does not match the number of epochs!")
    exit(1)
else:
    if len(desired_out_time_vector[0]) != num_instances:
        print("Error: Dimension of desired output time vector does not match the number of instances per epoch")
        exit(1)
    else:
        if len(desired_out_time_vector[0][0]["out_latency"]) != num_neurons_perLayer[-1]:
            print("Error: Dimension of desired output time vector does not match the number of output neurons")
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
neuron_idx = range(num_neurons)
synapse_addr = range(num_synapses)

## Initialize Connectivity Table and its "fan_in_synapse_idx", "fan_out_synapse_idx" attributes 
ConnectivityTable = SNN.ConnectivityInfo(num_neurons=num_neurons, 
                                         max_num_connections=max(num_neurons_perLayer), 
                                         max_num_fires=max_num_fires
                                        )
last_layer_last_synapse = -1
for layer in range (num_layers):
    for neuron in range(num_neurons_perLayer[layer]):
        # if (ConnectivityTable.neuron_idx[TableIdx] != TableIdx):
        #     print("TableIdx {0} does not match neuron_idx {1} in the Connectivity Table!"
        #             .format(TableIdx, ConnectivityTable.neuron_idx[TableIdx]))
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
# weight_vector = [initial_weight] * num_synapses
# weight_vector = [
#                     10, 10,
#                     8, 8, 10, 0, 0, 10, 8, 8,
#                     10, 10, 10, 10, 10, 10, 10, 10
#                 ]  

weight_vector = [
                    10, 10,
                    15, 15, 10, -15, -15, 10, 15, 15,
                    10, 10, 10, 10, 10, 10, 10, 10
                ] 

# weight_vector = [
#                     10, 10,
#                     15, 15, 10, -15, -15, 10, 15, 15,
#                     5, -4, -4, 5, 3, 2, 2, 3
#                 ]     # one set that works under threshold 120, output pattern [80,150]

# weight_vector = [
#                     10, 10,
#                     15, 15, 10, -15, -15, 10, 15, 15,
#                     4, -4, -3, 4, 3, 2, 2, 3
#                 ]    # another set that works under threshold 120, output pattern [80,150]


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
                                num_epochs=num_epochs
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
        output_layer_idx = 0
        for i in neuron_idx:
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
            # chekc if neuron in in the input layer
            if sn.layer_idx == 0:
                sn.training_on = 0
            # check if neuron is in the output layer 
            if sn.layer_idx == num_layers - 1:
                sn.training_on = 1
                sn.supervised = 1
                sn.spike_out_time_d_list =  [
                                                [
                                                    [desired_out_time_vector[epoch][instance]["out_latency"][output_layer_idx]]
                                                    for instance in range(num_instances)
                                                ] for epoch in range(num_epochs)
                                            ]
                output_layer_idx += 1

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
first_to_fire_class =  [
                            [None for instance in range(num_instances)]
                            for epoch in range(num_epochs)
                        ]

inference_correct =     [
                            [    
                                None for instance in range(num_instances) 
                            ] for epoch in range(num_epochs)
                        ]   # num_epochs x num_instances


temporal_diff =         [
                            [    
                                None for instance in range(num_instances) 
                            ] for epoch in range(num_epochs)
                        ]   # num_epochs x num_instances x num_output_neurons
                            # out-spike time - desired spike time 

temporal_diff_stats =   [
                            [    
                                [None] * num_neurons_perLayer[-1] 
                            ] for epoch in range(num_epochs)
                        ]   # num_epochs x num_output_neurons


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
            
            for i in neuron_idx:
                # check if the neuron being updated is in the output layer
                sn_list[epoch][instance][i].accumulate(sim_point=sim_point, 
                                        spike_in_info=spike_info[epoch][instance][sim_point], 
                                        WeightRAM_inst=WeightRAM,
                                        debug_mode=debug_mode,
                                        epoch=epoch,
                                        instance=instance,
                                        f_handle=f_handle,
                                        successive_correct_cnt=correct_cnt,
                                        coarse_fine_cut_off=stop_num*coarse_fine_ratio
                                        )                               
                # upadate the current potential to PotentialRAM
                PotentialRAM.potential[epoch][instance][i] = sn_list[epoch][instance][i].v[sim_point]
                
                # update the list of synapses that fired at this sim_point
                if (sn_list[epoch][instance][i].fire_cnt != -1):
                    if (sn_list[epoch][instance][i].spike_out_info[sn_list[epoch][instance][i].fire_cnt]["time"] == sim_point):
                        if (len(sn_list[epoch][instance][i].fan_out_synapse_addr) == 1): # if its an inner-layer neuron
                            fired_synapse_list[epoch][instance][sim_point].append(val)
                            spike_info[epoch][instance][sim_point]["fired_synapse_addr"].append(val)
                        else:   # if it's an output layer neuron that has fired
                            for key,val in enumerate(sn_list[epoch][instance][i].fan_out_synapse_addr):
                                fired_synapse_list[epoch][instance][sim_point].append(val)
                                spike_info[epoch][instance][sim_point]["fired_synapse_addr"].append(val)
                        
                        fired_neuron_list[epoch][instance][sim_point].append(sn_list[epoch][instance][i].neuron_idx)
                        # update first_to_fire_neuron during after this training instance
                        if (sn_list[epoch][instance][i].layer_idx == num_layers-1): 
                            if (first_to_fire_class[epoch][instance] == None):    
                                first_to_fire_class[epoch][instance] = \
                                    sn_list[epoch][instance][i].neuron_idx - sum(num_neurons_perLayer[0:-1])
                                temporal_diff[epoch][instance] = sim_point - \
                                    desired_out_time_vector[epoch][instance]["out_latency"][first_to_fire_class[epoch][instance]]
                            else:   # there are classes that fire at the same time
                                first_to_fire_class[epoch][instance] = num_neurons_perLayer[-1]
            
            # if a output neuron has fired once
            if first_to_fire_class[epoch][instance] != None:
                break

        # append statistics at the end of the training instance
        if first_to_fire_class[epoch][instance] == desired_out_time_vector[epoch][instance]["class_num"]:
            inference_correct[epoch][instance] = 1
            correct_cnt += 1
        else:
            inference_correct[epoch][instance] = 0
            correct_cnt = 0
        f_handle.write("Succesive correct count: {}\n".format(correct_cnt))
        f_handle.write("-------------------------------------------------\n")

        if correct_cnt == stop_num:
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
    plotNeuronResponse_iterative(sn_list=sn_list, epochs_list=[num_epochs-1], instance_list = [num_instances-1], only_output_layer=0)
    plt.show()
print("End of Program!")
