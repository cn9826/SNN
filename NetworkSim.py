import SNN
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from pandas import DataFrame



#%% Defined Functions 
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
    
def plotNeuronResponse_iterative(sn_list, epochs_list, only_output_layer=1):
    for epoch in epochs_list:    
        if only_output_layer == 0:
            for i in range(len(sn_list[epoch])):
                plotNeuronResponse(sn_list[epoch][i])
        else:
            for i in range(len(sn_list[epoch])):
                if sn_list[epoch][i].layer_idx == num_layers - 1:
                    plotNeuronResponse(sn_list[epoch][i])

#%% Initializations of Tables and Objects and Lists of Objects
#####################################################################################################################
## Specify Global Connectivity Parmeters
num_neurons_perLayer = [3,2,1]                           # Assuming num_neurons_perLayer is the number of connections in FC case
max_num_fires = 1

## Specify common Spiking Neuron Parameters
duration = 200
tau_u = 16      # in units with respect to duration
tau_v = None     # in units with respect to duration
threshold = 120

## Create stimulus spikes at the inuput layer (layer 0)
# Dimension: num_epochs x num_input_neurons
stimulus_time_vector = [
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80],
                        [0, 50, 80]                       
                       ]

## Specify desired out-spike times of the last-layer neurons
# 2-d array for now since max_num_fires is defaulted to 1
# row represents each training instance
# col represents the desired out-spike time for every output layer neuron 
num_epochs = 10
desired_out_time_list = [[None]*num_neurons_perLayer[-1] for rows in range(num_epochs)]
desired_out_time_list = \
    [
        [100],
        [100],
        [100],
        [100],
        [100],
        [100],
        [100],
        [100],
        [100],
        [100]
    ]

## Initilize synaptic weight for all the synapses
initial_weight = 8

## Simulation Settings
debug_mode = 1
plot_response = 1
#####################################################################################################################
if len(stimulus_time_vector) != num_epochs:
    print("Error: Dimension of stimulus does not match the number of epochs!")
    exit(1)
else:
    for vector in stimulus_time_vector:
        if len(vector) != num_neurons_perLayer[0]:
            print("Error: Number of stimulus does not match the number of neurons in the input layer!")
            exit(1)

stimulus_vector_info = [[]*num_neurons_perLayer[0] for epoch in range(num_epochs)]
for epoch in range(num_epochs):
    for synapse_idx in range(num_neurons_perLayer[0]):
        stimulus_entry = {}
        stimulus_entry["fan_in_synapse_addr"] = synapse_idx
        stimulus_entry["time"] = stimulus_time_vector[epoch][synapse_idx]
        stimulus_vector_info[epoch].append(stimulus_entry)

if len(desired_out_time_list) != num_epochs:
    print("desired_out_time_list row dimension does not match num_epochs")
    exit(1)
else:
    for rows in range(num_epochs):
        if len(desired_out_time_list[rows]) != num_neurons_perLayer[-1]:
            print(
                "desired_out_time_list column dimension does not the number of neurons in the last layer")
            exit(1)
            

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
weight_vector = [initial_weight] * num_synapses
WeightRAM = SNN.WeightRAM(num_synapses)
for i in synapse_addr:
    neuron_idx_WRAM, connection_num = index_2d(ConnectivityTable.fan_in_synapse_addr,synapse_addr[i])
    num_fan_in = len([fan_in for fan_in in ConnectivityTable.fan_in_synapse_addr[neuron_idx_WRAM] if fan_in is not None])    
    weight_vector[i] = int(weight_vector[i] / num_fan_in)
    WeightRAM.neuron_idx[i] = neuron_idx_WRAM
    WeightRAM.weight[i] = weight_vector[i]


## Initialize PotentialRAM --  a class with list attibutes sorted/indexed by neuron_idx; its contents are membrane potential & fan_out_synapse_addr
PotentialRAM = SNN.PotentialRAM(
                                num_neurons=num_neurons, 
                                max_num_connections=max(num_neurons_perLayer),
                                num_epochs=num_epochs
                               )
PotentialRAM.fan_out_synapse_addr = ConnectivityTable.fan_out_synapse_addr


## initialize a list of all SpikingNeuron objects
## and specify the desired out-spike time for output-layer neurons
# row represents each training instance (epoch)
# col represents each Spiking Neuron object sorted by ascending index 
sn_list = [[]*num_neurons for epoch in range(num_epochs)]
for epoch in range(num_epochs):
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
                                )
        # check if neuron is in the output layer 
        if sn.layer_idx == num_layers - 1:
            sn.training_on = 1
            sn.spike_out_time_d_list = [row[output_layer_idx] for row in desired_out_time_list]

        sn_list[epoch].append(sn)
#%% Simulation of SNN in time-major order

## Processing spikes in the network
fired_synapse_list = [[[] for row in range(0, sn.duration, sn.dt)] for epoch in range(num_epochs)]
fired_neuron_list = [[[] for row in range(0, sn.duration, sn.dt)] for epoch in range(num_epochs)]
                
spike_info = [
                [
                    {
                        "fired_synapse_addr": [],
                        "time"              : None
                    }
                    for row in range(0, sn.duration, sn.dt)
                ] for epoch in range(num_epochs)
             ]            
             # a list of dictionaries sorted by time steps

# for epoch in range(num_epochs):
for epoch in range(num_epochs):
    print("********************************Beginning of Epoch {}!***********************".format(epoch))
    for sim_point in range(0, sn.duration, sn.dt):
        # first check if any input synpase fires at this time step
        if sim_point in stimulus_time_vector[epoch]:
            fired_synapse_list[epoch][sim_point].append(stimulus_time_vector[epoch].index(sim_point))
            spike_info[epoch][sim_point]["fired_synapse_addr"].append(stimulus_time_vector[epoch].index(sim_point))

        spike_info[epoch][sim_point]["time"] = sim_point
        
        for i in neuron_idx:
            # check if the neuron being updated is in the output layer
            sn_list[epoch][i].accumulate(sim_point=sim_point, 
                                    spike_in_info=spike_info[epoch][sim_point], 
                                    WeightRAM_inst=WeightRAM,
                                    debug_mode=debug_mode,
                                    epoch=epoch
                                    )                               
            # upadate the current potential to PotentialRAM
            PotentialRAM.potential[epoch][i] = sn_list[epoch][i].v[sim_point]
            
            # update the list of synapses that fired during at this sim_point
            if (sn_list[epoch][i].fire_cnt != -1):
                if (sn_list[epoch][i].spike_out_info[sn_list[epoch][i].fire_cnt]["time"] == sim_point):
                    if (len(sn_list[epoch][i].fan_out_synapse_addr) == 1):
                        fired_synapse_list[epoch][sim_point].append(val)
                        spike_info[epoch][sim_point]["fired_synapse_addr"].append(val)
                    else:
                        for key,val in enumerate(sn_list[epoch][i].fan_out_synapse_addr):
                            fired_synapse_list[epoch][sim_point].append(val)
                            spike_info[epoch][sim_point]["fired_synapse_addr"].append(val)
                    
                    fired_neuron_list[epoch][sim_point].append(sn_list[epoch][i].neuron_idx)
    print("********************************End of Epoch {}!***********************\n\n".format(epoch))

print("End of Program!")

if plot_response:
    plotNeuronResponse_iterative(sn_list=sn_list, epochs_list=[0,num_layers-1], only_output_layer=0)
    plt.show()