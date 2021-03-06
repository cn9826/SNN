import numpy as np
import math


def list_intersection(lst1, lst2):
    if (isinstance(lst1, type(None)) or isinstance(lst2, type(None))):
        return []
    else:
        common_elements = list(set(lst1) & set(lst2))
        if None in common_elements:
            common_elements.remove(None)
        return(common_elements)


class ConnectivityInfo:
    def __init__(self, num_neurons, max_num_connections, max_num_fires):
        self.neuron_idx = range(num_neurons)
        self.layer_num = [None for row in range(num_neurons)]
        self.fan_in_synapse_addr = [[None for col in range(
            max_num_connections)] for row in range(num_neurons)]
        self.fan_out_synapse_addr = [[None for col in range(
            max_num_connections)] for row in range(num_neurons)]
        # self.fan_in_synapse_addr = np.empty((num_neurons, max_num_connections), dtype=int)
        # self.fan_out_synapse_addr = np.empty((num_neurons, max_num_connections),dtype=int)
        self.fire_cnt = [0 for row in range(num_neurons)]
        self.spike_out_time = [[None for col in range(
            max_num_fires)] for row in range(num_neurons)]
        # self.v = [0 for row in range(num_neurons)]


class WeightRAM:   # indexed by fan_in_synapse_addr
    def __init__(self, num_synapses):
        self.synapse_addr = range(num_synapses)
        self.weight = [None for row in range(num_synapses)]
        self.neuron_idx = [None for row in range(num_synapses)]


class PotentialRAM:  # indexed by neuron_idx
    def __init__(self, num_neurons, max_num_connections, num_epochs=1):
        self.neuron_idx = range(num_neurons) 
        self.potential =    [  
                                [
                                    0 for neurons in range(num_neurons) 
                                ] for epochs in range(num_epochs)  
                            ]
        self.fan_out_synapse_addr = [
                                        [None for col in range(max_num_connections)] 
                                        for row in range(num_neurons)    
                                    ]


class SpikingNeuron:   # this class can be viewed as the functional unit that updates neuron states
    # shared Class Variables
    # time step resolution w.r.t. duration
    dt = 1
    # Specific Object Variables

    # Later add num_connections, preNeuron_idx, synaptic_weights etc...
    def __init__(self, layer_idx, neuron_idx, fan_in_synapse_addr, fan_out_synapse_addr, tau_u, tau_v, 
                threshold, duration, spike_out_time_d_list=[], max_num_fires=1, training_on=0):
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.fan_in_synapse_addr = fan_in_synapse_addr
        # synapse address and weight is key-value pair
        self.fan_out_synapse_addr = fan_out_synapse_addr
        self.fire_cnt = -1
        # simulation duration specified in a.u.
        self.duration = duration
        self.u = [0] * int(round(self.duration/SpikingNeuron.dt))
        self.v = [0] * int(round(self.duration/SpikingNeuron.dt))
        self.tau_u = tau_u                  # current decay constant, in units of time step
        self.tau_v = tau_v                  # potential decay constant, in units of time step
        self.threshold = threshold          # potential threshold
        self.spike_out_time_d_list = spike_out_time_d_list    # a list of size 1 x num_epochs
        self.last_spike_in_info = []        # the last in-spike that contributed to the output spike
        self.oldWeight = []                 # the weight(s) that are associated with the last in-spike that causes the out-spike 
        self.spike_out_info = []            # output spike time info
                                            # a list of dictionaries with keys ("fan_out_synpase_addr", "time")
        self.max_num_fires = max_num_fires  # the maximum number a neuron can fire
        self.training_on = training_on
        self.relavent_fan_in_addr = []

    def fetchWeight(self, WeightRAM_inst, fired_synapse_addr):
        # fired_in_synapse_addr is a list of integer(s)
        return [WeightRAM_inst.weight[i] for i in fired_synapse_addr]

    def updateWeight(self, WeightRAM_inst, newWeight):
        if (len(self.relavent_fan_in_addr) != len(newWeight)):
            print(
                "Error: length of relavent_fan_in_addr {} does not match length of newWeight vector {}!"
                .format(len(self.relavent_fan_in_addr), len(newWeight)))
            exit(1)
        for idx, update_addr in enumerate(self.relavent_fan_in_addr):
            # find matching fan-in addresses to update
            matched_addr = WeightRAM_inst.synapse_addr.index(update_addr)
            WeightRAM_inst.weight[matched_addr] = newWeight[idx]

    # right now ReSuMe_training is only handling one desired out-spike time per output neuron 
    def ReSuMe_training(self, sim_point, spike_in_time, spike_out_time, epoch, 
                        spike_out_time_d, oldWeight,
                        kernel="exponential",
                        a_d=0, A_di=8, tau=16, debug=1):
        # applied only on the snypatic weights attached to neurons in the output layer

        # a_d is the non-Hebbian term to adjust the average strength of the synaptic input
        # so as to impose on a neuron a desired level of activity

        # spike_in_time is the most recent input synaptic spike that triggers an output spike
        # at spike_out_time

        # spike_out_time is the output spike time

        # spike_out_time_d is the desired firing time step (an integer)

        # oldWeight is the weight associated with the fan-in synapse that triggers the output spike at
        # spike_out_time

        # kernel pertains to the a_di term in ReSuMe Eq.2.10; it is default to exponential, can be changed
        # into other decaying kernel later

        # A_di is the coefficient of the kernel function a_di(s)

        # tau is the decay time constant

        # if output spike is ahead of desired output spike

        def computeHebbianTermUpdate(spike_out_time, spike_in_time, kernel, a_d, A_di, tau):
            if kernel != "exponential":
                print(
                    "Warning: only supports exponential kernel in ReSuMe learning right now!")
                print("Kernel is changed to exponential by default")
                kernel = "exponential"
            if kernel == "exponential":
                deltaWeight_Hebbian = A_di * \
                    math.exp(-(spike_out_time-spike_in_time)/tau)
            return int(deltaWeight_Hebbian)

        newWeight = [None]*len(oldWeight)
        # check if update is necessary
        if spike_out_time != spike_out_time_d:
            for i in range(len(newWeight)):
            
            ## update upon out-spike
                if sim_point == spike_out_time:
                    deltaWeight = 0
                    deltaWeight_Hebbian = computeHebbianTermUpdate(spike_out_time, spike_in_time,
                                                                    kernel, a_d, A_di, tau)
                    deltaWeight = -(a_d + deltaWeight_Hebbian)
                    newWeight[i] = oldWeight[i] + deltaWeight
                    if debug:
                        print("Epoch {}: Updated oldWeight: {} to newWeight: {} on Neuron {} upon out-spike at time {}"
                                .format(epoch, oldWeight[i], newWeight[i], self.neuron_idx, sim_point))
           
            ## update upon desired-spike
                elif sim_point == spike_out_time_d:
                    deltaWeight = 0
                    deltaWeight_Hebbian = computeHebbianTermUpdate(spike_out_time_d, spike_in_time,
                                                                    kernel, a_d, A_di, tau)
                    deltaWeight = +(a_d + deltaWeight_Hebbian)
                    newWeight[i] = oldWeight[i] + deltaWeight
                    if debug:
                        print("Epoch {}: Updated oldWeight: {} to newWeight: {} on Neuron {} upon desired spike at time {}"
                                .format(epoch, oldWeight[i], newWeight[i], self.neuron_idx, sim_point))
                else:
                    print("Epoch {}: Warning: During training on neuron {}, spike_out_time {} and spike_out_time_d {} has unclear relationship"
                            .format(epoch, self.neuron_idx, spike_out_time, spike_out_time_d))  
        else:
            newWeight = oldWeight[:]
            if debug:
                print("Epoch {}: Neuron {} fired at exactly the desired spike time {}"
                        .format(epoch, self.neuron_idx, sim_point))
        
        return newWeight         
        


    def accumulate(self, sim_point, spike_in_info, WeightRAM_inst, epoch, debug_mode=0, 
                   kernel="exponential", a_d=0, A_di=4, tau=16):     
        # spike_in_info is the data transmitted between neurons: (a dictionary)
        #   spike_in_info["fired_synapse_addr"] (a list of int)
        #   spike_in_info["time"] (an int)
        #   simpoint is in range(0,duration,dt)         
        
        dt = SpikingNeuron.dt

        # check if training is turned on and if training input parameters are valid
        if self.training_on == 1:
            if (isinstance(self.spike_out_time_d_list, list)==False): 
                print("Error: SpikingNeuron.accumulate() method has incorrect arguments!")
                print("Training is turned on but input paremeters are not specified correctly!")
                exit(1)

        # update synaptic current 
        # if sim_point == spike_in_info["time"] and fan-in matches, then process the spiking event
        if sim_point == spike_in_info["time"]:
            relavent_fan_in_addr = \
                                list_intersection(
                                                spike_in_info["fired_synapse_addr"], 
                                                self.fan_in_synapse_addr
                                                )
            if len(relavent_fan_in_addr) > 0:
                weight = SpikingNeuron.fetchWeight( self, 
                                                    WeightRAM_inst, 
                                                    fired_synapse_addr = relavent_fan_in_addr
                                                )                    # weight could potentially be a list of integers
                                                                        # if processing multiple fan-in spikes at one sim_point
    
                # when processing a in-spike event, keep a record of the spike_in_info 
                # and the associated weights for weight updates
                self.relavent_fan_in_addr = relavent_fan_in_addr
                self.last_spike_in_info = spike_in_info 
                self.oldWeight = weight

                self.u[sim_point] = (1 - dt/self.tau_u) * self.u[sim_point-1] + sum(weight) * dt
            else:
                if (self.u[sim_point-1] != 0):                          # added to take advantage of potential sparsity
                    self.u[sim_point] = (1 - dt/self.tau_u) * self.u[sim_point-1]
                else:
                    pass

        else:                                                       # if no spiking event, check sparsity 
            if (self.u[sim_point-1] != 0):                          # added to take advantage of potential sparsity
                self.u[sim_point] = (1 - dt/self.tau_u) * self.u[sim_point-1]
            else:
                pass
        # update membrane potential
        # check if neuron has reached maximaly allowed fire number
        if (self.fire_cnt < self.max_num_fires-1):
            if self.tau_v is not None:
                self.v[sim_point] = (1-dt/self.tau_v) * self.v[sim_point-1] + self.u[sim_point]*dt
            else:
                self.v[sim_point] = self.v[sim_point-1] + self.u[sim_point]*dt

            # check if spiked
            if (self.v[sim_point] >= self.threshold):
                self.v[sim_point] = 0
                spike_out_entry = {}
                spike_out_entry["fan_out_synapse_addr"] = self.fan_out_synapse_addr
                spike_out_entry["time"] = sim_point
                self.spike_out_info.append(spike_out_entry)
                self.fire_cnt += 1
                if debug_mode:
                    print("Epoch {}: Neuron {} at Layer{} has fired {} times at step {}"
                        .format(epoch, self.neuron_idx, self.layer_idx,self.fire_cnt + 1, [entry["time"] for entry in self.spike_out_info]))
            
                # if training is turned on, update synaptic weight 
                if self.training_on == 1:
                    spike_out_time_d = self.spike_out_time_d_list[epoch]
                    newWeight = SpikingNeuron.ReSuMe_training(self, sim_point=sim_point,
                                                spike_in_time=self.last_spike_in_info["time"],
                                                spike_out_time = self.spike_out_info[0]["time"],    # assume max_spike_num = 1
                                                spike_out_time_d = spike_out_time_d,
                                                epoch = epoch,
                                                oldWeight = self.oldWeight,
                                                kernel=kernel, a_d=a_d, A_di=A_di, tau=tau
                                                )
                    SpikingNeuron.updateWeight(self, WeightRAM_inst, newWeight)
                    self.oldWeight = newWeight[:]
        
        # update synaptic weight upon desired spike
        if self.training_on == 1:
            spike_out_time_d = self.spike_out_time_d_list[epoch]

            if sim_point == spike_out_time_d:    
                # check if self.last_spike_in_info is empty;
                # if yes then indicates the output-layer neuron has not even recieved inputs yet
                if len(self.last_spike_in_info) == 0:
                    print("Epoch {}: Error when training Neuron {} at time {}; the desired spike time precedes its inputs!"
                        .format(epoch, sim_point, self.neuron_idx))
                    exit(1)

                # check if the neuron has fired
                if self.fire_cnt != -1:
                    newWeight = SpikingNeuron.ReSuMe_training(self, sim_point=sim_point,
                                                spike_in_time=self.last_spike_in_info["time"],
                                                spike_out_time =self.spike_out_info[0]["time"],     # assume max_spike_num = 1
                                                spike_out_time_d = spike_out_time_d,
                                                epoch = epoch,
                                                oldWeight = self.oldWeight,
                                                kernel=kernel, a_d=a_d, A_di=A_di, tau=tau
                                                )
                else:
                    newWeight = SpikingNeuron.ReSuMe_training(self, sim_point=sim_point,
                                                spike_in_time=self.last_spike_in_info["time"],
                                                spike_out_time=None,     
                                                spike_out_time_d = spike_out_time_d,
                                                epoch = epoch,
                                                oldWeight = self.oldWeight,
                                                kernel=kernel, a_d=a_d, A_di=A_di, tau=tau
                                                )

                SpikingNeuron.updateWeight(self, WeightRAM_inst, newWeight)
                self.oldWeight = newWeight[:]
            
                    

                    
                
