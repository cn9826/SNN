import numpy as np
import math

#%% Utility Functions
def list_intersection(lst1, lst2):
    if (isinstance(lst1, type(None)) or isinstance(lst2, type(None))):
        return []
    # else:
    #     common_elements = list(set(lst1) & set(lst2))
    #     if None in common_elements:
    #         common_elements.remove(None)
    else:
        common_elements = [element_lst1 for element_lst1 in lst1 if element_lst1 in lst2]
        common_elements_indices_lst1 = [i for i in range(len(lst1)) if lst1[i] in lst2]
        if len(common_elements) != len(common_elements_indices_lst1):
            print("Error when calling list_intersection(): len(common_elements) is not equal to len(common_elements_indices_lst1)")
            exit(1)
        return(common_elements, common_elements_indices_lst1)

def clip_newWeight (newWeight, max_weight, min_weight):
    if newWeight > max_weight:
        newWeight = max_weight
    elif newWeight < min_weight:
        newWeight = min_weight
    return newWeight

#%% Class Definitions
class ConnectivityInfo:
    def __init__(self, num_neurons):
        self.neuron_idx = range(num_neurons)
        self.layer_num = [None for row in range(num_neurons)]
        self.fan_out_neuron_idx = [[] for row in range(num_neurons)]
        self.fan_in_neuron_idx = [[] for row in range(num_neurons)]
        self.fan_in_synapse_addr = [ [] for row in range(num_neurons)]
        self.fan_out_synapse_addr = [ [] for row in range(num_neurons)]
        self.num_fan_in_neurons_established = [ 0 for row in range(num_neurons)]


class WeightRAM:   # indexed by fan_in_synapse_addr
    def __init__(self, num_synapses):
        self.synapse_addr = range(num_synapses)
        self.weight = [None for row in range(num_synapses)]
        self.pre_neuron_idx = [None for row in range(num_synapses)]
        self.post_neuron_idx = [None for row in range(num_synapses)]

class PotentialRAM:  # indexed by neuron_idx
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neuron_idx = range(num_neurons)
        self.potential = [0 for neurons in range(num_neurons)]
        self.fan_out_synapse_addr = [
                                        []
                                        for row in range(num_neurons)
                                    ]
    def clearPotential(self):
        self.potential = [0 for neurons in range(self.num_neurons)]

class SpikeIncache:
    def __init__(self, depth_causal=4, depth_anticausal=2):
        self.depth_causal = depth_causal
        self.depth_anticausal = depth_anticausal
        self.depth = depth_causal + depth_anticausal
        self.causal_spike_in_cnt = 0
        self.write_ptr = 0
        self.mem = [
            {
                "fired_synapse_addr": None,
                "causal_tag": None,
                "weight": None,
                "time": None
            } for entry in range(self.depth)
        ]

    def writeSpikeInInfo(self, fired_synapse_addr, time, weight):
        if self.write_ptr < self.depth:
            # write first
            self.mem[self.write_ptr]["fired_synapse_addr"] = fired_synapse_addr
            self.mem[self.write_ptr]["time"] = time
            self.mem[self.write_ptr]["weight"] = weight
            self.mem[self.write_ptr]["causal_tag"] = 0
            if self.causal_spike_in_cnt < self.depth_causal:
                self.causal_spike_in_cnt += 1
                self.mem[self.write_ptr]["causal_tag"] = 1
            # then increment write_ptr
            self.write_ptr += 1

    def writeNewWeight(self, causal_fan_in_addr, newWeight):
        if len(causal_fan_in_addr) != len(newWeight):
            print(
                "Error when calling SpikeInCache.writeNewWeight(): len(causal_fan_in_addr) {} does not correspond to len(newWeight) {}!"
                .format(len(causal_fan_in_addr), len(newWeight)))
            exit(1)
        for i in range(len(causal_fan_in_addr)):
            for j in range(len(self.mem)):
                if self.mem[j]["fired_synapse_addr"] == causal_fan_in_addr[i]:
                    self.mem[j]["weight"] = newWeight[i]
                    break

    def clearMem(self):
        self.causal_spike_in_cnt = 0
        self.write_ptr = 0
        self.mem = [
            {
                "fired_synapse_addr": None,
                "causal_tag": None,
                "weight": None,
                "time": None
            } for entry in range(self.depth)
        ]

class SpikeIncache_Hidden:
    def __init__(self, depth_causal=2, depth_anticausal=2):
        self.depth_causal = depth_causal
        self.depth_anticausal = depth_anticausal
        self.depth = depth_causal + depth_anticausal
        self.write_ptr = 0
        self.fired = 0
        self.mem = [
            {
                "fired_synapse_addr": None,
                "causal_tag": None,
                "weight": None,
                "time": None
            } for entry in range(self.depth)
        ]

    def writeSpikeInInfo(self, fired_synapse_addr, time, weight):
        if self.write_ptr < self.depth:
            # check if the causal entry has been filled
            if self.write_ptr < self.depth_causal:
                if not self.fired:
                    # write first
                    self.mem[self.write_ptr]["fired_synapse_addr"] = fired_synapse_addr
                    self.mem[self.write_ptr]["time"] = time
                    self.mem[self.write_ptr]["weight"] = weight
                    self.mem[self.write_ptr]["causal_tag"] = 1
                    # then increment write_ptr
                    self.write_ptr += 1
                else:
                    self.write_ptr = self.depth_causal
                    self.mem[self.write_ptr]["fired_synapse_addr"] = fired_synapse_addr
                    self.mem[self.write_ptr]["time"] = time
                    self.mem[self.write_ptr]["weight"] = weight
                    self.mem[self.write_ptr]["causal_tag"] = 0
                    self.write_ptr += 1

            # different from SpikeIncache for output layer neurons, SpikeIncache for hidden layer
            # draws a causal/anti-causal cutoff depending on whether this hidden neuron has fired or not
            else:
                # only going to record in-spike event as anti-causal if the hidden neuron
                # has already fired
                if self.fired:
                    self.mem[self.write_ptr]["fired_synapse_addr"] = fired_synapse_addr
                    self.mem[self.write_ptr]["time"] = time
                    self.mem[self.write_ptr]["weight"] = weight
                    self.mem[self.write_ptr]["causal_tag"] = 0
                    self.write_ptr += 1

    def clearMem(self):
        self.write_ptr = 0
        self.fired = 0
        self.mem = [
            {
                "fired_synapse_addr": None,
                "causal_tag": None,
                "weight": None,
                "time": None
            } for entry in range(self.depth)
        ]

class SpikeIncache_Output:
    def __init__(self, num_sublocations=3, depth_causal_per_subloc=2, depth_anticausal_per_subloc=2):
        self.num_sublocations = num_sublocations
        self.depth_causal_per_subloc = depth_causal_per_subloc
        self.depth_anticausal_per_subloc = depth_anticausal_per_subloc
        self.depth_per_subloc = depth_causal_per_subloc + depth_anticausal_per_subloc
        self.write_ptr = [0 for sublocation in range(self.num_sublocations)]
        self.mem =  \
                [
                    [
                        {
                            "fired_synapse_addr"    :   None,
                            "sublocation_idx"       :   None,
                            "causal_tag"            :   None,
                            "weight"                :   None,
                            "time"                  :   None
                        } for entry in range(self.depth_per_subloc)
                    ] for sublocation in range(self.num_sublocations)
                ]
        self.sublocation_buffer = [None for sublocation in range(self.num_sublocations)]
        self.sublocation_buffer_ptr = 0
        self.sublocation_buffer_prev = [None for sublocation in range(self.num_sublocations)]
        self.fired = 0

    # references sublocation_buffer_prev
    def writeSpikeInInfo(self, fired_synapse_addr, sublocation_idx, time, weight):
        # at the beginning of recording location-specific in-spike events
        if self.sublocation_buffer_ptr == 0:
            # record the sublocation_idx in the sublocation_buffer if it has not been recorded before
            if not sublocation_idx in self.sublocation_buffer_prev:
                self.writeSublocationBuffer(sublocation_idx)
                # then record the in-spike event in the mem
                self.mem[0][self.write_ptr[0]]["fired_synapse_addr"] = fired_synapse_addr
                self.mem[0][self.write_ptr[0]]["sublocation_idx"] = sublocation_idx
                self.mem[0][self.write_ptr[0]]["causal_tag"] = 1
                self.mem[0][self.write_ptr[0]]["weight"] = weight
                self.mem[0][self.write_ptr[0]]["time"] = time
                self.write_ptr[0] += 1

        # if there has been sublocation_idx registered during this instance
        else:
            # check if the sublocation has been registered in sublocation_buffer_prev
            if not sublocation_idx in self.sublocation_buffer_prev:
                # if sublocation_idx has been registered in this instance
                if sublocation_idx in self.sublocation_buffer:
                    buffer_idx = self.sublocation_buffer.index(sublocation_idx)
                    self.registerInSpikeEvents(buffer_idx, fired_synapse_addr, sublocation_idx, time, weight)
                # if sublocation_idx has NOT been registered in this instance
                else:
                    if self.sublocation_buffer_ptr < self.num_sublocations:
                        buffer_idx = self.sublocation_buffer_ptr
                        self.writeSublocationBuffer(sublocation_idx)
                        self.registerInSpikeEvents(buffer_idx, fired_synapse_addr, sublocation_idx, time, weight)

    # Does NOT reference sublocation_buffer_prev
    def writeSpikeInInfo_loose(self, fired_synapse_addr, sublocation_idx, time, weight):
        # at the beginning of recording location-specific in-spike events
        if self.sublocation_buffer_ptr == 0:
            self.writeSublocationBuffer(sublocation_idx)
            # then record the in-spike event in the mem
            self.mem[0][self.write_ptr[0]]["fired_synapse_addr"] = fired_synapse_addr
            self.mem[0][self.write_ptr[0]]["sublocation_idx"] = sublocation_idx
            self.mem[0][self.write_ptr[0]]["causal_tag"] = 1
            self.mem[0][self.write_ptr[0]]["weight"] = weight
            self.mem[0][self.write_ptr[0]]["time"] = time
            self.write_ptr[0] += 1
        else:
            # if sublocation_idx has been registered in this instance
            if sublocation_idx in self.sublocation_buffer:
                buffer_idx = self.sublocation_buffer.index(sublocation_idx)
                self.registerInSpikeEvents(buffer_idx, fired_synapse_addr, sublocation_idx,
                                           time, weight)
            # if sublocation_idx has NOT been registered in this instance
            else:
                if self.sublocation_buffer_ptr < self.num_sublocations:
                    buffer_idx = self.sublocation_buffer_ptr
                    self.writeSublocationBuffer(sublocation_idx)
                    self.registerInSpikeEvents(buffer_idx, fired_synapse_addr, sublocation_idx,
                                               time, weight)


    def writeSublocationBuffer(self, sublocation_idx):
        self.sublocation_buffer[self.sublocation_buffer_ptr] = sublocation_idx
        self.sublocation_buffer_ptr += 1

    def registerInSpikeEvents(self, buffer_idx, fired_synapse_addr, sublocation_idx, time, weight):
        # if the causal event depth has not been filled
        if self.write_ptr[buffer_idx] < self.depth_causal_per_subloc:
            # check if the output neuron has spiked
            if not self.fired:
                self.mem[buffer_idx][self.write_ptr[buffer_idx]]["causal_tag"] = 1
            else:
                self.write_ptr[buffer_idx] = self.depth_causal_per_subloc
                self.mem[buffer_idx][self.write_ptr[buffer_idx]]["causal_tag"] = 0
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["fired_synapse_addr"] = fired_synapse_addr
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["sublocation_idx"] = sublocation_idx
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["weight"] = weight
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["time"] = time
            if self.write_ptr[buffer_idx] != self.depth_per_subloc-1:
                self.write_ptr[buffer_idx] += 1
        # if write_ptr[buffer_idx] has traversed to the anticausal depth
        else:
            if not self.fired:
                self.mem[buffer_idx][self.write_ptr[buffer_idx]]["causal_tag"] = 1
            else:
                self.mem[buffer_idx][self.write_ptr[buffer_idx]]["causal_tag"] = 0
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["fired_synapse_addr"] = fired_synapse_addr
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["sublocation_idx"] = sublocation_idx
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["weight"] = weight
            self.mem[buffer_idx][self.write_ptr[buffer_idx]]["time"] = time
            if self.write_ptr[buffer_idx] == self.depth_per_subloc-1:
                self.write_ptr[buffer_idx] = self.depth_causal_per_subloc
            else:
                self.write_ptr[buffer_idx] += 1

    def clearMem(self):
        self.write_ptr = [0 for sublocation in range(self.num_sublocations)]
        self.sublocation_buffer_ptr = 0
        self.sublocation_buffer = [None for sublocation in range(self.num_sublocations)]
        self.fired = 0
        self.mem =  \
                [
                    [
                        {
                            "fired_synapse_addr"    :   None,
                            "sublocation_idx"       :   None,
                            "causal_tag"            :   None,
                            "weight"                :   None,
                            "time"                  :   None
                        } for entry in range(self.depth_per_subloc)
                    ] for sublocation in range(self.num_sublocations)
                ]


    def latchSublocationBufferPrev(self):
        self.sublocation_buffer_prev = self.sublocation_buffer

class SpikingNeuron:
    dt = 1                  # tick
    decaying_current = 1    # turn on decaying current neuron dynamics

    def __init__(self, layer_idx, neuron_idx, sublocation_idx, fan_in_synapse_addr, fan_out_synapse_addr, tau_u, tau_v,
                threshold, duration, depth_causal, depth_anticausal, num_sublocations=3, spike_out_time_d_list=[],
                training_on=0, supervised=0):
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.fan_in_synapse_addr = fan_in_synapse_addr
        # synapse address and weight is key-value pair
        self.fan_out_synapse_addr = fan_out_synapse_addr
        self.fire_cnt = -1
        self.sublocation_idx = sublocation_idx
        # simulation duration specified in a.u.
        self.duration = duration

        self.u = [0] * int(round(self.duration / SpikingNeuron.dt))
        self.tau_u = tau_u  # current decay constant, in units of time step
        self.v = [0] * int(round(self.duration / SpikingNeuron.dt))
        self.tau_v = tau_v  # potential decay constant, in units of time step

        self.threshold = threshold  # potential threshold
        self.spike_out_time_d_list = spike_out_time_d_list
        # a list of size num_instances x 1
        # only used when ReSuMe is used for training
        if layer_idx == 0:
            self.spike_in_cache = SpikeIncache(depth_causal=depth_causal, depth_anticausal=depth_anticausal)
        elif layer_idx == 1:
            self.spike_in_cache = SpikeIncache_Hidden(depth_causal=depth_causal, depth_anticausal=depth_anticausal)
        elif layer_idx == 2:
            self.spike_in_cache = SpikeIncache_Output(num_sublocations=num_sublocations,
                                                      depth_causal_per_subloc=depth_causal,
                                                      depth_anticausal_per_subloc=depth_anticausal)

        if layer_idx != 2:
            self.spike_out_info = \
                [
                    {
                        "fired_synapse_addr": synapse_index,
                        "time": None,
                        "sublocation_idx": self.sublocation_idx
                    } for synapse_index in fan_out_synapse_addr
                ]
        else:
            self.spike_out_info = \
                [
                    {
                        "fired_synapse_addr": None,
                        "time": None,
                        "sublocation_idx": self.sublocation_idx
                    }
                ]

        self.training_on = training_on
        self.supervised = supervised

    def fetchWeight(self, WeightRAM_inst, fired_synapse_addr):
        # fired_in_synapse_addr is a list of integer(s)
        return [WeightRAM_inst.weight[i] for i in fired_synapse_addr]

    def updateWeight(self, fan_in_addr, WeightRAM_inst, newWeight):
        if (not isinstance(fan_in_addr, list)) or (not isinstance(newWeight, list)):
            print("Error when calling \"updateWeight\": both fan_in_addr {} and newWeight {} need to be lists !"
                  .format(fan_in_addr, newWeight))
            exit(1)
        # find matching fan-in addresses to update
        for i in range(len(fan_in_addr)):
            matched_addr = WeightRAM_inst.synapse_addr.index(fan_in_addr[i])
            WeightRAM_inst.weight[matched_addr] = newWeight[i]

    def findSynapseGroup(self, instance, f_handle, intended_output, num_causal, num_anticausal, output_or_hidden,
                        hidden_causal_reverse_search=1, hidden_anticausal_reverse_search=0, debug=1):
        if not output_or_hidden in ["output", "hidden"]:
            print("Error when calling SpikingNeruon.findSynapseGroup(): argument \"output_or_hidden\" {} is not sepcified correctly!"
                .format(output_or_hidden))
            exit(1)
        if output_or_hidden == "output":
            ## used on output layer neurons to find causal and anti-causal in-spike events based on which to update its
            ## synaptic weights and trace presynaptic hidden neurons

            ## num_causal specify how many causal in-spike events to update on and always searched from the top-of-the-queue in each sublocation mem block
            ## num_anticausal specifies how many anti-causal in-spike events to update on and always searched from the end-of-the-queue in each sublocation mem block

            ## look for causal in-spike events in ascending order from the start-of-the-queue
            found_cnt_causal = [0 for x in range(self.spike_in_cache.num_sublocations)]
            in_spike_events_causal = []
            for buffer_idx in range(self.spike_in_cache.sublocation_buffer_ptr):
                for i in range(0, self.spike_in_cache.depth_causal_per_subloc):
                    if (self.spike_in_cache.mem[buffer_idx][i]["time"] != None
                            and self.spike_in_cache.mem[buffer_idx][i]["causal_tag"] == 1):
                        found_cnt_causal[buffer_idx] += 1
                        in_spike_events_causal.append(self.spike_in_cache.mem[buffer_idx][i])
                    if found_cnt_causal[buffer_idx] == num_causal \
                            or self.spike_in_cache.mem[buffer_idx][i]["causal_tag"] == 0:
                        break
                # if found fewer causal in-spike events than the designated num_causal
                # append subsequent in-spike events as causal anyway to meet num_causal quota
                if found_cnt_causal[buffer_idx] < num_causal:
                    if intended_output:
                        print(
                            "Instance {}: Neuron {} has found {} causal SpikeInCache entries on sublocation {}, less than specified {}"
                            .format(instance, self.neuron_idx, found_cnt_causal[buffer_idx],
                                    self.spike_in_cache.sublocation_buffer[buffer_idx], num_causal))
                    if debug:
                        f_handle.write(
                            "Instance {}: Neuron {} has found {} causal SpikeInCache entries on sublocation {}, less than specified {}\n"
                            .format(instance, self.neuron_idx, found_cnt_causal[buffer_idx],
                                    self.spike_in_cache.sublocation_buffer[buffer_idx], num_causal))

            ## look for anti-causal in-spike events in descending order from the end-of-the-queue
            found_cnt_anticausal = [0 for x in range(self.spike_in_cache.num_sublocations)]
            in_spike_events_anticausal = []

            # look for anticausal in-spike events in descending order from the end-of-the-queue
            for buffer_idx in range(self.spike_in_cache.sublocation_buffer_ptr):

                for i in range(self.spike_in_cache.depth_per_subloc - 1,
                               self.spike_in_cache.depth_causal_per_subloc - 1, -1):
                    if (self.spike_in_cache.mem[buffer_idx][i]["time"] != None
                            and self.spike_in_cache.mem[buffer_idx][i]["causal_tag"] == 0):
                        found_cnt_anticausal[buffer_idx] += 1
                        in_spike_events_anticausal.append(self.spike_in_cache.mem[buffer_idx][i])
                    if found_cnt_anticausal[buffer_idx] == num_anticausal \
                            or self.spike_in_cache.mem[buffer_idx][i]["causal_tag"] == 1:
                        break

                # if found fewer anti-causal in-spike events than the designated num_anticausal
                if found_cnt_anticausal[buffer_idx] < num_anticausal:
                    # # append preceding in-spike events as anti-causal anyway to meet num_anticausal quota
                    # num_diff = num_anticausal - found_cnt_anticausal[buffer_idx]
                    # in_spike_events_anticausal.extend(
                    #     [
                    #         self.spike_in_cache.mem[buffer_idx][idx] for idx in range(i - num_diff + 1, i + 1)
                    #         if self.spike_in_cache.mem[buffer_idx][idx]["time"] != None
                    #     ]
                    # )
                    if intended_output:
                        print(
                            "Instance {}: Neuron {} has found {} anti-causal SpikeInCache entries on sublocation {}, less than specified {}"
                            .format(instance, self.neuron_idx, found_cnt_anticausal[buffer_idx],
                                    self.spike_in_cache.sublocation_buffer[buffer_idx], num_anticausal))
                    if debug:
                        f_handle.write(
                            "Instance {}: Neuron {} has found {} anti-causal SpikeInCache entries on sublocation {}, less than specified {}\n"
                            .format(instance, self.neuron_idx, found_cnt_anticausal[buffer_idx],
                                    self.spike_in_cache.sublocation_buffer[buffer_idx], num_anticausal))

        elif output_or_hidden == "hidden":
            # first look for the first mem_idx that has a "causal_tag" of 0
            anticausal_starting_idx = None
            for i in range(self.spike_in_cache.depth):
                if (self.spike_in_cache.mem[i]["time"]!=None
                    and self.spike_in_cache.mem[i]["causal_tag"] == 0):
                    anticausal_starting_idx = i
                    break
            if anticausal_starting_idx == None:
                anticausal_starting_idx = self.spike_in_cache.depth
                print("Instance {}: Hidden Neuron {} could not find an anti-causal in-spike entry!"
                    .format(instance, self.neuron_idx))
                if debug:
                    f_handle.write("Instance {}: Hidden Neuron {} could not find an anti-causal in-spike entry!\n"
                        .format(instance, self.neuron_idx))

            ## look for causal in-spike events
            found_cnt_causal = 0
            in_spike_events_causal = []
            if hidden_causal_reverse_search:
                for i in range(anticausal_starting_idx - 1, -1, -1):
                    if (self.spike_in_cache.mem[i]["time"] != None
                            and self.spike_in_cache.mem[i]["causal_tag"] == 1):
                        in_spike_events_causal.append(self.spike_in_cache.mem[i])
                        found_cnt_causal += 1
                    if found_cnt_causal == num_causal:
                        break
            elif not hidden_causal_reverse_search:
                for i in range(0, anticausal_starting_idx, 1):
                    if (self.spike_in_cache.mem[i]["time"] != None
                            and self.spike_in_cache.mem[i]["causal_tag"] == 1):
                        in_spike_events_causal.append(self.spike_in_cache.mem[i])
                        found_cnt_causal += 1
                    if found_cnt_causal == num_causal:
                        break
            if found_cnt_causal < num_causal:
                print("Instance {}: Hidden Neuron {} has found {} causal SpikeInCache entries, less than specified {}"
                      .format(instance, self.neuron_idx, found_cnt_causal, num_causal))
                if debug:
                    f_handle.write(
                        "Instance {}: Hidden Neuron {} has found {} causal SpikeInCache entries, less than specified {}\n"
                        .format(instance, self.neuron_idx, found_cnt_causal, num_causal))

            ## look for anti-causal in-spike events
            found_cnt_anticausal = 0
            in_spike_events_anticausal = []
            if not hidden_anticausal_reverse_search:
                for i in range(anticausal_starting_idx, self.spike_in_cache.depth, 1):
                    if (self.spike_in_cache.mem[i]["time"]!=None
                        and self.spike_in_cache.mem[i]["causal_tag"] == 0):
                        in_spike_events_anticausal.append(self.spike_in_cache.mem[i])
                        found_cnt_anticausal += 1
                    if found_cnt_anticausal == num_anticausal:
                        break
            elif hidden_anticausal_reverse_search:
                for i in range(self.spike_in_cache.depth-1, anticausal_starting_idx-1, -1):
                    if (self.spike_in_cache.mem[i]["time"]!=None
                        and self.spike_in_cache.mem[i]["causal_tag"] == 0):
                        in_spike_events_anticausal.append(self.spike_in_cache.mem[i])
                        found_cnt_anticausal += 1
                    if found_cnt_anticausal == num_anticausal:
                        break
            if found_cnt_anticausal < num_anticausal:
                print("Instance {}: Hidden Neuron {} has found {} anti-causal SpikeInCache entries, less than specified {}"
                    .format(instance, self.neuron_idx, found_cnt_anticausal, num_anticausal))
                if debug:
                    f_handle.write("Instance {}: Hidden Neuron {} has found {} anti-causal SpikeInCache entries, less than specified {}\n"
                    .format(instance, self.neuron_idx, found_cnt_anticausal, num_anticausal))

        return (in_spike_events_causal, in_spike_events_anticausal)

    def findPreSynapticNeuron(self, fan_in_synapse_addr, WeightRAM_inst):
        # fan_in_synapse_addr is an int
        return [WeightRAM_inst.pre_neuron_idx[addr] for addr in fan_in_synapse_addr]

    def RC_output_subloc_specific(self, t_out, t_min, instance,
                    oldWeight_causal, causal_fan_in_addr, t_in_causal, tag_causal,
                    oldWeight_anticausal, anticausal_fan_in_addr, t_in_anticausal, tag_anticausal,
                    f_handle,
                    reward_signal, isf2f, isIntended,
                    moving_accuracy, accuracy_th,
                    A_causal_coarse=2, A_causal_fine=1, tau_causal=50,
                    A_anticausal_coarse=2, A_anticausal_fine=1, tau_anticausal=30,
                    max_weight=7, min_weight=-8, deltaWeight_causal_default=2, deltaWeight_anticausal_default=1,
                    debug=0):
        ## Determine A
        if moving_accuracy >= accuracy_th:
            A_causal = A_causal_fine
            A_anticausal = A_anticausal_fine
            if debug:
                f_handle.write("Instance {}: switching to Fine-update at out-spike time {}\n"
                               .format(instance, t_out))
        else:
            A_causal = A_causal_coarse
            A_anticausal = A_anticausal_coarse

        newWeight_causal = oldWeight_causal[:]
        if anticausal_fan_in_addr != None:
            newWeight_anticausal = oldWeight_anticausal[:]
        else:
            newWeight_anticausal = None

        if isIntended:
            # F2F P+ on the intended, t_ref = t_in
            if reward_signal and isf2f:
                for i in range(len(oldWeight_causal)):
                    if tag_causal[i] == 1:
                        s_causal = t_out - t_in_causal[i]
                        deltaWeight_causal = \
                            round(A_causal * math.exp(-(s_causal / tau_causal)))
                    else:
                        deltaWeight_causal = deltaWeight_causal_default
                    newWeight_causal[i] = oldWeight_causal[i] + deltaWeight_causal
                    newWeight_causal[i] = clip_newWeight(newWeight_causal[i], max_weight, min_weight)
                    if debug:
                        f_handle.write(
                            "Instance {}: Causal      F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                            .format(instance, oldWeight_causal[i], newWeight_causal[i], causal_fan_in_addr[i],
                                    self.neuron_idx, t_out))

                if anticausal_fan_in_addr != None:
                    for i in range(len(oldWeight_anticausal)):
                        if tag_anticausal[i] == 0:
                            s_anticausal = t_out - t_in_anticausal[i]
                            deltaWeight_anticausal = \
                                round(-A_anticausal * math.exp(s_anticausal / tau_anticausal))
                        else:
                            deltaWeight_anticausal = -deltaWeight_anticausal_default
                        newWeight_anticausal[i] = oldWeight_anticausal[i] + deltaWeight_anticausal
                        newWeight_anticausal[i] = clip_newWeight(newWeight_anticausal[i], max_weight, min_weight)
                        if debug:
                            f_handle.write(
                                "Instance {}: anti-Causal F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                                .format(instance, oldWeight_anticausal[i], newWeight_anticausal[i],
                                        anticausal_fan_in_addr[i], self.neuron_idx, t_out))

            # non-F2F P- on the intended, t_ref = t_min
            elif not reward_signal and not isf2f:
                if t_out != None:
                    s = t_out - t_min
                    deltaWeight_causal = \
                        round(A_causal * math.exp(-(s / tau_causal)))
                    if anticausal_fan_in_addr != None:
                        deltaWeight_anticausal = \
                            round(-A_anticausal * math.exp(s / tau_anticausal))
                else:
                    deltaWeight_causal = deltaWeight_causal_default
                    if anticausal_fan_in_addr != None:
                        deltaWeight_anticausal = -deltaWeight_anticausal_default

                for i in range(len(oldWeight_causal)):
                    newWeight_causal[i] = oldWeight_causal[i] + deltaWeight_causal
                    newWeight_causal[i] = clip_newWeight(newWeight_causal[i], max_weight, min_weight)
                    if debug:
                        f_handle.write(
                            "Instance {}: Causal      non-F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                            .format(instance, oldWeight_causal[i], newWeight_causal[i], causal_fan_in_addr[i],
                                    self.neuron_idx, t_out))

                if anticausal_fan_in_addr != None:
                    for i in range(len(oldWeight_anticausal)):
                        newWeight_anticausal[i] = oldWeight_anticausal[i] + deltaWeight_anticausal
                        newWeight_anticausal[i] = clip_newWeight(newWeight_anticausal[i], max_weight, min_weight)
                        if debug:
                            f_handle.write(
                                "Instance {}: anti-Causal      non-F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                                .format(instance, oldWeight_anticausal[i], newWeight_anticausal[i],
                                        anticausal_fan_in_addr[i], self.neuron_idx, t_out))

        elif not isIntended:
            # non-F2F P+ on the non-intended, t_ref = t_min
            if reward_signal and not isf2f:
                s = t_out - t_min
                deltaWeight_causal = \
                    round(-A_causal * math.exp(-s / tau_causal))
                if anticausal_fan_in_addr != None:
                    deltaWeight_anticausal = \
                        round(A_anticausal * math.exp(s / tau_anticausal))

                for i in range(len(oldWeight_causal)):
                    newWeight_causal[i] = oldWeight_causal[i] + deltaWeight_causal
                    newWeight_causal[i] = clip_newWeight(newWeight_causal[i], max_weight, min_weight)
                    if debug:
                        f_handle.write(
                            "Instance {}: Causal      non-F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                            .format(instance, oldWeight_causal[i], newWeight_causal[i], causal_fan_in_addr[i],
                                    self.neuron_idx, t_out))

            # F2F P- on the non-intended, t_ref = t_in
            elif not reward_signal and isf2f:
                for i in range(len(oldWeight_causal)):
                    if tag_causal[i] == 1:
                        s_causal = t_out - t_in_causal[i]
                        deltaWeight_causal = \
                            round(-A_causal * math.exp(-(s_causal / tau_causal)))
                    else:
                        deltaWeight_causal = -deltaWeight_causal_default
                    newWeight_causal[i] = oldWeight_causal[i] + deltaWeight_causal
                    newWeight_causal[i] = clip_newWeight(newWeight_causal[i], max_weight, min_weight)
                    if debug:
                        f_handle.write(
                            "Instance {}: Causal      F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                            .format(instance, oldWeight_causal[i], newWeight_causal[i], causal_fan_in_addr[i],
                                    self.neuron_idx, t_out))

        return(newWeight_causal, newWeight_anticausal)


    def RSTDP_hidden(self, spike_in_time, spike_out_time, instance,
                    oldWeight, fan_in_addr, synapse_causal_tag, neuron_causal_tag, f_handle,
                    reward_signal, isf2f, isIntended,
                    moving_accuracy, accuracy_th,
                    kernel="constant",
                    A_causal_coarse=2, A_causal_fine=1, tau_causal=30,
                    A_anticausal_coarse=2, A_anticausal_fine=1, tau_anticausal=30,
                    A_const_coarse=2, A_const_fine=1,
                    max_weight=7, min_weight=0,
                    debug=0):

        # oldWeight and newWeight are lists of integers
        # synapse_causal_tag is a list of "causal_tag" field of the SpikeIncache entry corresponding to oldWeight
        # fan_in_addr is a list of fan-in synapse addresses corresponding to oldWeight
        # spike_in_time is a list of in-spike timings corresponding to fan_in_addr
        # spike_out_time should be specified as an int

        def computeCausalCutoff(t_out, t_in, synapse_causal_tag, neuron_causal_tag, isIntended,
                                A_causal, tau_causal, A_anticausal, tau_anticausal, A_const,
                                kernel=kernel, default_deltaWeight=1):
            if kernel == "constant":
                if synapse_causal_tag == 1:
                    deltaWeight = A_const
                elif synapse_causal_tag == 0:
                    deltaWeight = -A_const
            elif kernel == "exponential":
                if t_out == None:
                    if synapse_causal_tag == 1:
                        deltaWeight = default_deltaWeight
                    elif synapse_causal_tag == 0:
                        deltaWeight = -1*default_deltaWeight
                else:
                    s = t_out - t_in
                    if synapse_causal_tag == 1:
                        deltaWeight = \
                            A_causal * math.exp(-s/tau_causal)
                    elif synapse_causal_tag == 0:
                        deltaWeight = \
                            -A_anticausal * math.exp(s/tau_anticausal)

            if (isIntended ^ neuron_causal_tag):
                deltaWeight = -1*deltaWeight
            return round(deltaWeight)


        kernel_list = ["constant", "exponential"]

        if not kernel in kernel_list:
            print("Error when calling SpikingNeuron.RSTDP_hidden(): kernel {} is not specified correctly!"
                .format(kernel))
            exit(1)
        if not isIntended:
            print(
                "Error when calling SpikingNeuron.RSTDP_hidden(): called when isIntended != 1!"
                 )
            exit(1)
        if len(fan_in_addr) != len(oldWeight):
            print("Error when calling SpikingNeuron.RSTDP_hidden(): len(fan_in_addr) {} does not correspond to len(oldWeight) {}!"
                .format(len(fan_in_addr), len(oldWeight)))
            exit(1)
        if len(spike_in_time) != len(oldWeight):
            print("Error when calling SpikingNeuron.RSTDP_hidden(): len(spike_in_time) {} does not correspond to len(oldWeight) {}!"
                .format(len(spike_in_time), len(oldWeight)))
            exit(1)
        if (not isinstance(spike_out_time, int)) and spike_out_time != None:
            print("Error when calling SpikingNeuron.RSTDP_hidden(): spike_out_time {} should be an integer or None!"
                .format(spike_out_time))
            exit(1)


        if neuron_causal_tag:
            neuron_causal_str = "Causal       "
        elif not neuron_causal_tag:
            neuron_causal_str = "anti-Causal  "

        if moving_accuracy >= accuracy_th:
            A_causal = A_causal_fine
            A_anticausal = A_anticausal_fine
            A_const = A_const_fine
        else:
            A_causal = A_causal_coarse
            A_anticausal = A_anticausal_coarse
            A_const = A_const_coarse

        newWeight = [None] * len(oldWeight)
        for i in range(len(oldWeight)):
            # F2F P+ and Non-F2F P- on the intended
            deltaWeight = computeCausalCutoff(t_out=spike_out_time,
                                                t_in=spike_in_time[i],
                                                synapse_causal_tag=synapse_causal_tag[i],
                                                neuron_causal_tag=neuron_causal_tag,
                                                isIntended=isIntended,
                                                A_causal=A_causal, tau_causal=tau_causal,
                                                A_anticausal=A_anticausal, tau_anticausal=tau_anticausal,
                                                A_const=A_const
                                                )
            newWeight[i] = oldWeight[i] + deltaWeight
            newWeight[i] = clip_newWeight(newWeight=newWeight[i], max_weight=max_weight, min_weight=min_weight)
            if debug:
                if reward_signal and isf2f:
                    f_handle.write("Instance {}: F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on {} Hidden Neuron {} upon out-spike at time {}\n"
                    .format(instance, oldWeight[i], newWeight[i], fan_in_addr[i], neuron_causal_str, self.neuron_idx, spike_out_time))
                elif (not reward_signal) and (not isf2f):
                    f_handle.write("Instance {}: Non-F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on {} Hidden Neuron {} upon out-spike at time {}\n"
                    .format(instance, oldWeight[i], newWeight[i], fan_in_addr[i], neuron_causal_str, self.neuron_idx, spike_out_time))

        return newWeight

    def accumulate(self, sim_point, spike_in_info, WeightRAM_inst, instance, f_handle,
                   debug_mode=0
                   ):
        # spike_in_info is a list of dictionary
        #   spike_in_info["fired_synapse_addr"] (a list of int)
        #   spike_in_info["time"] (an int)
        #   simpoint is in range(0,duration,dt)

        dt = SpikingNeuron.dt

        # check for in-spike events to process
        relavent_fan_in_addr = []
        sublocation_idx_list = []
        for entry in spike_in_info:
            if entry["fired_synapse_addr"] in self.fan_in_synapse_addr:
                relavent_fan_in_addr.append(entry["fired_synapse_addr"])
                sublocation_idx_list.append(entry["sublocation_idx"])

        # if there are events to process
        if len(relavent_fan_in_addr) > 0:
            weight = SpikingNeuron.fetchWeight(self,
                                               WeightRAM_inst,
                                               fired_synapse_addr=relavent_fan_in_addr
                                               )  # weight could potentially be a list of integers

            if self.layer_idx != 2:
                for i in range(len(relavent_fan_in_addr)):
                    self.spike_in_cache.writeSpikeInInfo(
                        fired_synapse_addr=relavent_fan_in_addr[i],
                        time=sim_point,
                        weight=weight[i]
                    )
            else:
                for i in range(len(relavent_fan_in_addr)):
                    self.spike_in_cache.writeSpikeInInfo_loose(
                                        fired_synapse_addr=relavent_fan_in_addr[i],
                                        sublocation_idx=sublocation_idx_list[i],
                                        time=sim_point,
                                        weight = weight[i]
                                        )
            # if decaying current is turned on
            if SpikingNeuron.decaying_current:
                self.u[sim_point] = (1 - dt / self.tau_u) * self.u[sim_point - 1] + \
                                    sum(weight) * dt
                if (self.fire_cnt == -1):
                    if self.tau_v is not None:
                        self.v[sim_point] = (1 - dt / self.tau_v) * self.v[sim_point - 1] + self.u[sim_point] * dt
                    else:
                        self.v[sim_point] = self.v[sim_point - 1] + self.u[sim_point] * dt

            # if delta current is used then directly add weight onto potential
            else:
                if (self.fire_cnt == -1):
                    if self.tau_v is not None:
                        self.v[sim_point] = (1 - dt / self.tau_v) * self.v[sim_point - 1] + sum(weight)
                    else:
                        self.v[sim_point] = self.v[sim_point - 1] + sum(weight)

            # check if spiked
            if (self.v[sim_point] >= self.threshold):
                self.v[sim_point] = 0
                for entry in self.spike_out_info:
                    entry["time"] = sim_point
                self.fire_cnt += 1
                # notify hidden neuron's spike_in_cache that this neuron has fired
                if self.layer_idx == 1 or self.layer_idx == 2:
                    self.spike_in_cache.fired = 1
                if debug_mode:
                    # print("Instance {}: Neuron {} at Layer{} has fired {} times at step {}"
                    #     .format(instance, self.neuron_idx, self.layer_idx,self.fire_cnt + 1, sim_point))
                    f_handle.write("Instance {}: Neuron {} at Layer{} has fired {} times at step {}\n"
                                   .format(instance, self.neuron_idx, self.layer_idx, self.fire_cnt + 1, sim_point))

        # if no event to process
        else:
            if SpikingNeuron.decaying_current:
                if (self.u[sim_point - 1] != 0):  # added to take advantage of potential sparsity
                    self.u[sim_point] = (1 - dt / self.tau_u) * self.u[sim_point - 1]
                    if (self.fire_cnt == -1):
                        if self.tau_v is not None:
                            self.v[sim_point] = (1 - dt / self.tau_v) * self.v[sim_point - 1] + self.u[sim_point] * dt
                        else:
                            self.v[sim_point] = self.v[sim_point - 1] + self.u[sim_point] * dt
                        # check if spiked
                        if (self.v[sim_point] >= self.threshold):
                            self.v[sim_point] = 0
                            for entry in self.spike_out_info:
                                entry["time"] = sim_point
                            self.fire_cnt += 1
                            # notify hidden neuron's spike_in_cache that this neuron has fired
                            if self.layer_idx == 1 or self.layer_idx == 2:
                                self.spike_in_cache.fired = 1
                            if debug_mode:
                                # print("Instance {}: Neuron {} at Layer{} has fired {} times at step {}"
                                #     .format(instance, self.neuron_idx, self.layer_idx,self.fire_cnt + 1, sim_point))
                                f_handle.write("Instance {}: Neuron {} at Layer{} has fired {} times at step {}\n"
                                               .format(instance, self.neuron_idx, self.layer_idx, self.fire_cnt + 1,
                                                       sim_point))
            else:
                if self.tau_v is not None:
                    self.v[sim_point] = (1 - dt / self.tau_v) * self.v[sim_point - 1]
                else:
                    self.v[sim_point] = self.v[sim_point - 1]


    def clearStateVariables(self):
        self.u = [0] * int(round(self.duration / SpikingNeuron.dt))
        self.v = [0] * int(round(self.duration / SpikingNeuron.dt))

        self.fire_cnt = -1
        for entry in self.spike_out_info:
            entry["time"] = None
        self.spike_in_cache.clearMem()

def combined_RSTDP_BRRC(sn_list, instance, inference_correct, num_fired_output,
                        supervised_hidden, supervised_output, f_handle,
                        PreSynapticIdx_intended, PreSynapticIdx_nonintended,
                        desired_ff_idx, min_fire_time,
                        f2f_neuron_lst, non_f2f_neuron_lst, f2f_neuron_idx,
                        WeightRAM, moving_accuracy, accuracy_th, correct_cnt,
                        num_causal_output=2, num_anticausal_output=1,
                        num_causal_hidden=6, num_anticausal_hidden=6, debug_mode=0
                        ):
    # expect num_fired_output = len(output_neuron_fire_info[instance][neuron_idx])
    # desired_ff_idx = desired_ff_neuron[instance]["ff_neuron"]
    # f2f_neuron_idx should be a single int
    def hiddenNeuronUpdateRoutine(sn_hidden, instance, reward_signal, isf2f, isIntended,
                                neuron_causal_tag, f_handle, WeightRAM,
                                moving_accuracy, accuracy_th,
                                num_causal_hidden=num_causal_hidden,
                                num_anticausal_hidden=num_anticausal_hidden,
                                debug_mode=debug_mode):
        spike_out_time = sn_hidden.spike_out_info[0]["time"]

        in_spike_events_causal, in_spike_events_anticausal = \
            sn_hidden.findSynapseGroup(instance=instance, f_handle=f_handle,
                                       intended_output=0,
                                       num_causal=num_causal_hidden,
                                       num_anticausal=num_anticausal_hidden,
                                       output_or_hidden="hidden",
                                       debug=debug_mode)
        spike_in_time = \
            [entry["time"] for entry in in_spike_events_causal] + \
            [entry["time"] for entry in in_spike_events_anticausal]

        oldWeight = \
            [entry["weight"] for entry in in_spike_events_causal] + \
            [entry["weight"] for entry in in_spike_events_anticausal]

        fan_in_addr = \
            [entry["fired_synapse_addr"] for entry in in_spike_events_causal] + \
            [entry["fired_synapse_addr"] for entry in in_spike_events_anticausal]

        synapse_causal_tag = \
            [entry["causal_tag"] for entry in in_spike_events_causal] + \
            [entry["causal_tag"] for entry in in_spike_events_anticausal]

        newWeight = sn_hidden.RSTDP_hidden(spike_in_time=spike_in_time, spike_out_time=spike_out_time,
                                        instance=instance, oldWeight=oldWeight,
                                        fan_in_addr=fan_in_addr,
                                        synapse_causal_tag=synapse_causal_tag,
                                        neuron_causal_tag=neuron_causal_tag,
                                        f_handle=f_handle,
                                        reward_signal=reward_signal, isf2f=isf2f, isIntended=isIntended,
                                        moving_accuracy=moving_accuracy, accuracy_th=accuracy_th,
                                        debug=debug_mode
                                        )
        sn_hidden.updateWeight(fan_in_addr=fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight)

    def intendedUpdateRoutine(sn_intended, supervised_hidden, reward_signal, isIntended, isf2f,
                                instance, min_fire_time, f_handle,
                                PreSynapticIdx_intended, WeightRAM,
                                moving_accuracy, accuracy_th, output_silent=0,
                                num_causal_output=num_causal_output,
                                num_anticausal_output=num_anticausal_output,
                                num_causal_hidden=num_causal_hidden,
                                num_anticausal_hidden=num_anticausal_hidden,
                                disparate_learning_on=0, debug_mode=debug_mode
                                ):
        if not isIntended:
            print("Error when calling intendedUpdateRoutine(): function argument isIntended is {}!".format(isIntended))
            exit(1)

        in_spike_events_causal, in_spike_events_anticausal = \
            sn_intended.findSynapseGroup(instance=instance, f_handle=f_handle,
                                         intended_output=1,
                                         num_causal=num_causal_output,
                                         num_anticausal=num_anticausal_output,
                                         output_or_hidden="output", debug=debug_mode)

        causal_fan_in_addr = [entry["fired_synapse_addr"] for entry in in_spike_events_causal]
        t_in_causal = [entry["time"] for entry in in_spike_events_causal]
        oldWeight_causal = [entry["weight"] for entry in in_spike_events_causal]
        tag_causal = [entry["causal_tag"] for entry in in_spike_events_causal]
        if len(in_spike_events_anticausal) > 0:
            anticausal_fan_in_addr = [entry["fired_synapse_addr"] for entry in in_spike_events_anticausal]
            t_in_anticausal = [entry["time"] for entry in in_spike_events_anticausal]
            oldWeight_anticausal = [entry["weight"] for entry in in_spike_events_anticausal]
            tag_anticausal = [entry["causal_tag"] for entry in in_spike_events_anticausal]
        else:
            anticausal_fan_in_addr = None
            t_in_anticausal = None
            oldWeight_anticausal = None
            tag_anticausal = None

        if sn_intended.fire_cnt == -1:
            t_out = None
        else:
            t_out = sn_intended.spike_out_info[0]["time"]

        if output_silent:
            anticausal_fan_in_addr = None
            t_in_anticausal = None
            oldWeight_anticausal = None
            tag_anticausal = None

        newWeight_causal, newWeight_anticausal = \
            sn_intended.RC_output_subloc_specific(
                                    t_out=t_out,
                                    t_min=min_fire_time,
                                    instance=instance,
                                    oldWeight_causal=oldWeight_causal,
                                    causal_fan_in_addr=causal_fan_in_addr,
                                    t_in_causal=t_in_causal, tag_causal=tag_causal,
                                    oldWeight_anticausal=oldWeight_anticausal,
                                    anticausal_fan_in_addr=anticausal_fan_in_addr,
                                    t_in_anticausal=t_in_anticausal, tag_anticausal=tag_anticausal,
                                    f_handle=f_handle,
                                    reward_signal=reward_signal,
                                    isf2f=isf2f, isIntended=isIntended,
                                    moving_accuracy=moving_accuracy, accuracy_th=accuracy_th,
                                    debug=debug_mode
                                    )

        sn_intended.updateWeight(fan_in_addr= causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight_causal)
        if anticausal_fan_in_addr != None:
            sn_intended.updateWeight(fan_in_addr=anticausal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight_anticausal)

        # trace back ahead to the presynaptic hidden neuron from the intended output neuron
        PreSynapticIdx_intended[instance]["causal"].extend(
            sn_intended.findPreSynapticNeuron(causal_fan_in_addr, WeightRAM)
            )
        for i in range(len(PreSynapticIdx_intended[instance]["causal"])):
            sn_hidden_causal = sn_list[PreSynapticIdx_intended[instance]["causal"][i]]
            if supervised_hidden:
                hiddenNeuronUpdateRoutine(sn_hidden=sn_hidden_causal, instance=instance,
                                          reward_signal=reward_signal, isf2f=isf2f, isIntended=isIntended,
                                          neuron_causal_tag=1,
                                          f_handle=f_handle, WeightRAM=WeightRAM,
                                          moving_accuracy=moving_accuracy, accuracy_th=accuracy_th)

        # only if the output neuron can find an anticausal_fan_in_addr will the presynaptic neuron be traced
        if anticausal_fan_in_addr != None and disparate_learning_on:
            PreSynapticIdx_intended[instance]["anti-causal"].extend(
                sn_intended.findPreSynapticNeuron(anticausal_fan_in_addr, WeightRAM)
            )
            for i in range(len(PreSynapticIdx_intended[instance]["anti-causal"])):
                sn_hidden_anticausal = sn_list[PreSynapticIdx_intended[instance]["anti-causal"][i]]
                if supervised_hidden:
                    hiddenNeuronUpdateRoutine(sn_hidden=sn_hidden_anticausal, instance=instance,
                                              reward_signal=reward_signal, isf2f=isf2f, isIntended=isIntended,
                                              neuron_causal_tag=0,
                                              f_handle=f_handle, WeightRAM=WeightRAM,
                                              moving_accuracy=moving_accuracy, accuracy_th=accuracy_th)


    def nonintendedUpdateRoutine(sn_nonintended, supervised_hidden, reward_signal, isIntended, isf2f,
                                instance, min_fire_time, f_handle,
                                PreSynapticIdx_nonintended, WeightRAM,
                                moving_accuracy, accuracy_th,
                                num_causal_output=num_causal_output,
                                num_anticausal_output=num_anticausal_output,
                                debug_mode=debug_mode):
        if isIntended:
            print("Error when calling nonintendedUpdateRoutine(): function argument isIntended is {}!".format(isIntended))
            exit(1)

        in_spike_events_causal, in_spike_events_anticausal = \
            sn_nonintended.findSynapseGroup(instance=instance, f_handle=f_handle,
                                        intended_output=0,
                                         num_causal=num_causal_output,
                                         num_anticausal=num_anticausal_output,
                                         output_or_hidden="output", debug=debug_mode)
        causal_fan_in_addr = [entry["fired_synapse_addr"] for entry in in_spike_events_causal]
        t_in_causal = [entry["time"] for entry in in_spike_events_causal]
        oldWeight_causal = [entry["weight"] for entry in in_spike_events_causal]
        tag_causal = [entry["causal_tag"] for entry in in_spike_events_causal]

        t_out = sn_nonintended.spike_out_info[0]["time"]

        newWeight_causal, _ = \
            sn_nonintended.RC_output_subloc_specific(
                                    t_out=t_out,
                                    t_min=min_fire_time,
                                    instance=instance,
                                    oldWeight_causal=oldWeight_causal,
                                    causal_fan_in_addr=causal_fan_in_addr,
                                    t_in_causal=t_in_causal, tag_causal=tag_causal,
                                    oldWeight_anticausal=None,
                                    anticausal_fan_in_addr=None,
                                    t_in_anticausal=None, tag_anticausal=None,
                                    f_handle=f_handle,
                                    reward_signal=reward_signal,
                                    isf2f=isf2f, isIntended=isIntended,
                                    moving_accuracy=moving_accuracy, accuracy_th=accuracy_th,
                                    debug=debug_mode
                                    )
        sn_nonintended.updateWeight(fan_in_addr=causal_fan_in_addr, WeightRAM_inst=WeightRAM, newWeight=newWeight_causal)

    if num_fired_output > 0:
        if desired_ff_idx == f2f_neuron_idx:
            # check if there is any other neuron that fired at the same min_fire_time
            if len(f2f_neuron_lst) == 1:
                correct_cnt += 1
                inference_correct[instance] = 1
            elif len(f2f_neuron_lst) > 1:
                correct_cnt = 0
                inference_correct[instance] = 0

            ### Training On the Intended F2F neuron
            if supervised_output:
                # apply F2F P+ on the intended output neuron
                reward_signal = 1
                isIntended = 1
                isf2f = 1
                sn_intended = sn_list[desired_ff_idx]
                intendedUpdateRoutine(
                    sn_intended=sn_intended, supervised_hidden=supervised_hidden,
                    reward_signal=reward_signal, isIntended=isIntended, isf2f=isf2f,
                    instance=instance, min_fire_time=min_fire_time, f_handle=f_handle,
                    PreSynapticIdx_intended=PreSynapticIdx_intended, WeightRAM=WeightRAM,
                    moving_accuracy=moving_accuracy, accuracy_th=accuracy_th
                )

            ### Training On the non-intended non-F2F neurons
            if supervised_output:
                # apply non-F2F P+ on the non-intended output neuron that fired within separation_window
                reward_signal = 1
                isIntended = 0
                isf2f = 0
                if len(non_f2f_neuron_lst) > 0:
                    for non_f2f_idx in non_f2f_neuron_lst:
                        sn_nonintended = sn_list[non_f2f_idx]
                        nonintendedUpdateRoutine(
                            sn_nonintended=sn_nonintended, supervised_hidden=supervised_hidden,
                            reward_signal=reward_signal, isIntended=isIntended, isf2f=isf2f,
                            instance=instance, min_fire_time=min_fire_time, f_handle=f_handle,
                            PreSynapticIdx_nonintended=PreSynapticIdx_nonintended, WeightRAM=WeightRAM,
                            moving_accuracy=moving_accuracy, accuracy_th=accuracy_th
                        )

        elif desired_ff_idx != f2f_neuron_idx:
            correct_cnt = 0
            inference_correct[instance] = 0
            ### Training on the Intended non-F2F neuron
            if supervised_output:
                # apply non-F2F P- on the intended non-F2F output neuron
                reward_signal = 0
                isf2f = 0
                isIntended = 1
                sn_intended = sn_list[desired_ff_idx]
                intendedUpdateRoutine(
                    sn_intended=sn_intended, supervised_hidden=supervised_hidden,
                    reward_signal=reward_signal, isIntended=isIntended, isf2f=isf2f,
                    instance=instance, min_fire_time=min_fire_time, f_handle=f_handle,
                    PreSynapticIdx_intended=PreSynapticIdx_intended, WeightRAM=WeightRAM,
                    moving_accuracy=moving_accuracy, accuracy_th=accuracy_th
                )

                ### Training on the non-intended F2F neuron
            if supervised_output:
                # apply F2F P- on the non-intended F2F output neuron
                reward_signal = 0
                isIntended = 0
                isf2f = 1
                sn_nonintended = sn_list[f2f_neuron_idx]
                nonintendedUpdateRoutine(
                    sn_nonintended=sn_nonintended, supervised_hidden=supervised_hidden,
                    reward_signal=reward_signal, isIntended=isIntended, isf2f=isf2f,
                    instance=instance, min_fire_time=min_fire_time, f_handle=f_handle,
                    PreSynapticIdx_nonintended=PreSynapticIdx_nonintended, WeightRAM=WeightRAM,
                    moving_accuracy=moving_accuracy, accuracy_th=accuracy_th
                )


    # if none of the output layer neuron has fired
    else:
        correct_cnt = 0
        inference_correct[instance] = 0
        ## Strengthen the last 2 causal synaptic weights on the intended output-layer
        if supervised_output:
            reward_signal = 0
            isf2f = 0
            isIntended = 1
            sn_intended = sn_list[desired_ff_idx]
            intendedUpdateRoutine(
                sn_intended=sn_intended, supervised_hidden=supervised_hidden,
                reward_signal=reward_signal, isIntended=isIntended, isf2f=isf2f,
                instance=instance, min_fire_time=min_fire_time, f_handle=f_handle,
                PreSynapticIdx_intended=PreSynapticIdx_intended, WeightRAM=WeightRAM,
                moving_accuracy=moving_accuracy, accuracy_th=accuracy_th,
                output_silent=1
            )
        if debug_mode:
            f_handle.write("Instance {}: no output layer neuron has fired up until the end of forward pass\n"
            .format(instance))
        print("Instance {}: no output layer neuron has fired up until the end of forward pass!"
        .format(instance))

    return correct_cnt

