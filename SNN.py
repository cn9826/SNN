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
    def __init__(self, num_neurons, max_num_connections, num_instances=1):
        self.neuron_idx = range(num_neurons) 
        self.potential =    [  
                                [0 for neurons in range(num_neurons)] 
                                for instance in range(num_instances)
                            ]
        self.fan_out_synapse_addr = [
                                        [None for col in range(max_num_connections)] 
                                        for row in range(num_neurons)    
                                    ]

class SpikeIncache:
    def __init__(self, depth=8, ptr_offset = -4):
        self.depth = depth
        self.ptr_offset = ptr_offset
        self.write_ptr = 0
        self.mem =  [
                        {
                            "fired_synapse_addr"    :   None,
                            "weight"                :   None,
                            "time"                  :   None
                        } for entry in range(depth)
                    ]
    def writeSpikeInInfo(self, fired_synapse_addr, time, weight):
        # write first 
        self.mem[self.write_ptr]["fired_synapse_addr"] = fired_synapse_addr
        self.mem[self.write_ptr]["time"] = time
        self.mem[self.write_ptr]["weight"] = weight
        # then increment write_ptr
        if self.write_ptr == self.depth - 1:
            self.write_ptr = 0
        else:
            self.write_ptr += 1

    def getUpdateAddr(self, isf2f, reward_signal, isIntended):
        # to deal with non-F2F P- learning on the intended neuron
        if not isf2f and not reward_signal and isIntended: 
            mem_idx = (self.write_ptr + self.ptr_offset - 1) % self.depth
        else:
            mem_idx = self.write_ptr - 1
        return (self.mem[mem_idx]["fired_synapse_addr"], self.mem[mem_idx]["weight"], self.mem[mem_idx]["time"])

class SpikingNeuron:   # this class can be viewed as the functional unit that updates neuron states
    # shared Class Variables
    # time step resolution w.r.t. duration
    dt = 1
    # Specific Object Variables

    # Later add num_connections, preNeuron_idx, synaptic_weights etc...
    def __init__(self, layer_idx, neuron_idx, fan_in_synapse_addr, fan_out_synapse_addr, tau_u, tau_v, 
                threshold, duration, spike_out_time_d_list=[],
                max_num_fires=1, training_on=0, supervised=0):
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
        
        self.spike_out_time_d_list = spike_out_time_d_list    
                                            # a list of size num_instances x 1
                                            # only used when ReSuMe is used for training

        self.last_spike_in_info = []        # the last in-spike that contributed to the output spike
        self.causal_spike_in_info = []      # the causal in-spike info
        self.spike_in_cache = SpikeIncache()
        self.oldWeight = None               # the weight that are associated with the causal in-spike that causes the out-spike 
                                            # an int 
                                            
        self.spike_out_info = []            # output spike time info
                                            # a list of dictionaries with keys ("fan_out_synpase_addr", "time")
        self.max_num_fires = max_num_fires  # the maximum number a neuron can fire
        self.training_on = training_on
        self.supervised = supervised
        self.relavent_fan_in_addr = []
        self.causal_fan_in_addr = []        # the fan-in synpase addresses corresponding to a causal in-spike
                                            # a causal in-spike is defined as one that either triigers an out-spike
                                            # or one that preceeds a desired spike

    def fetchWeight(self, WeightRAM_inst, fired_synapse_addr):
        # fired_in_synapse_addr is a list of integer(s)
        return [WeightRAM_inst.weight[i] for i in fired_synapse_addr]

    def updateWeight(self, fan_in_addr, WeightRAM_inst, newWeight):
        if (not isinstance(fan_in_addr, int)) or (not isinstance(newWeight, int)):
            print("Error when calling \"updateWeight\": both fan_in_addr {} and newWeight {} needs to be integer!"
                  .format(fan_in_addr, newWeight))
            exit(1)
        # find matching fan-in addresses to update
        matched_addr = WeightRAM_inst.synapse_addr.index(fan_in_addr)
        WeightRAM_inst.weight[matched_addr] = newWeight




    # right now ReSuMe_training is only handling one desired out-spike time per output neuron 
    def ReSuMe_training(self, sim_point, spike_in_time, spike_out_time,instance,
                        spike_out_time_d, oldWeight, causal_fan_in_addr, f_handle, 
                        successive_correct_cnt,
                        coarse_fine_cut_off,  
                        kernel="exponential", 
                        a_d=0, A_di_coarse=8, tau_coarse=16, 
                        A_di_fine=1, tau_fine=4, 
                        debug=1):
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
                return round(deltaWeight_Hebbian)

        newWeight = [None]*len(oldWeight)
        # check if coarse update is necessary
        if (successive_correct_cnt < coarse_fine_cut_off 
            and spike_out_time != spike_out_time_d):
            for i in range(len(newWeight)):
            
            ## update upon out-spike
                if sim_point == spike_out_time:
                    deltaWeight = 0
                    deltaWeight_Hebbian = computeHebbianTermUpdate(spike_out_time, spike_in_time,
                                                                    kernel, a_d, A_di_coarse, tau_coarse)
                    deltaWeight = -(a_d + deltaWeight_Hebbian)
                    newWeight[i] = oldWeight[i] + deltaWeight
                    if debug:
                        f_handle.write("Instance {}: Updated oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                                .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, sim_point))
           
            ## update upon desired-spike
                elif sim_point == spike_out_time_d:
                    deltaWeight = 0
                    deltaWeight_Hebbian = computeHebbianTermUpdate(spike_out_time_d, spike_in_time,
                                                                    kernel, a_d, A_di_coarse, tau_coarse)
                    deltaWeight = +(a_d + deltaWeight_Hebbian)
                    newWeight[i] = oldWeight[i] + deltaWeight
                    if debug:
                        f_handle.write("Instance {}: Updated oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon desired spike at time {}\n"
                                .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, sim_point))
                else:
                    print("Instance{}: Warning: During training on neuron {}, spike_out_time {} and spike_out_time_d {} has unclear relationship"
                            .format(instance, self.neuron_idx, spike_out_time, spike_out_time_d))  
        
        # check if fine update is necessary
        elif (successive_correct_cnt >= coarse_fine_cut_off
                and spike_out_time != spike_out_time_d):
            for i in range(len(newWeight)):

            ## update upon out-spike
                if sim_point == spike_out_time:
                    deltaWeight = 0
                    deltaWeight_Hebbian = computeHebbianTermUpdate(spike_out_time, spike_in_time,
                                                                    kernel, a_d, A_di_fine, tau_fine)
                    deltaWeight = -(a_d + deltaWeight_Hebbian)
                    newWeight[i] = oldWeight[i] + deltaWeight
                    if debug:
                        f_handle.write("Instance {}: Fine-updated oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                                .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, sim_point))

            ## update upon desired-spike
                elif sim_point == spike_out_time_d:
                    deltaWeight = 0
                    deltaWeight_Hebbian = computeHebbianTermUpdate(spike_out_time_d, spike_in_time,
                                                                    kernel, a_d, A_di_fine, tau_fine)
                    deltaWeight = +(a_d + deltaWeight_Hebbian)
                    newWeight[i] = oldWeight[i] + deltaWeight
                    if debug:
                        f_handle.write("Instance {}: Fine-updated oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon desired spike at time {}\n"
                                .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, sim_point))
                else:
                    print("Instance{}: Warning: During training on neuron {}, spike_out_time {} and spike_out_time_d {} has unclear relationship"
                            .format(instance, self.neuron_idx, spike_out_time, spike_out_time_d))  

        elif (spike_out_time == spike_out_time_d):
            newWeight = oldWeight[:]
            if debug:
                f_handle.write("Instance{}: Neuron {} fired at exactly the desired spike time {}\n"
                        .format(instance, self.neuron_idx, sim_point))
        


        return newWeight         
        
    def STDP_training(self, sim_point, spike_in_time, spike_out_time, instance,
                        oldWeight, f_handle, kernel="rectangle", max_weight=15, min_weight=-15,
                        A_di=3, tau_pos=16, tau_neg=16, debug=1):
        
        kernel_list = ["exponential", "rectangle"]
        if not kernel in kernel_list:
            print("Error when calling SimpleSTDP_training: kernel {} is not recognized!".format(kernel))
            exit(1)

        newWeight = [None] * len(oldWeight)
        for i in range(len(newWeight)):
            if spike_in_time > spike_out_time:
                if kernel == "exponential":
                    deltaWeight =  -1 * round(A_di * math.exp((spike_out_time-spike_in_time)/tau_neg))
                elif kernel == "rectangle":
                    deltaWeight = -1 * A_di
                newWeight[i] = oldWeight[i] + deltaWeight
                if newWeight[i] > max_weight:
                    newWeight[i]=max_weight    
                elif newWeight[i] < min_weight:
                    newWeight[i]=min_weight
                if debug:
                    f_handle.write("Instance {}: anti-Causal Updated oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon late in-spike at time {}\n"
                            .format(instance, oldWeight[i], newWeight[i], self.relavent_fan_in_addr[i], self.neuron_idx, sim_point))
            
            elif spike_in_time <= spike_out_time:
                if kernel == "exponential":
                    deltaWeight = round(A_di * math.exp((spike_out_time-spike_in_time)/tau_neg))
                elif kernel == "rectangle":
                    deltaWeight = A_di
                newWeight[i] = oldWeight[i] + deltaWeight                
                if newWeight[i] > max_weight:
                    newWeight[i]=max_weight    
                elif newWeight[i] < min_weight:
                    newWeight[i]=min_weight
                if debug:
                    f_handle.write("Instance {}: Causal Updated oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                            .format(instance, oldWeight[i], newWeight[i], self.relavent_fan_in_addr[i], self.neuron_idx, sim_point))
            
        return newWeight                                

    
    def BinaryReward_training(self, spike_in_time, spike_out_time, instance,
                              oldWeight, causal_fan_in_addr, f_handle,
                              reward_signal, successive_correct_cnt, coarse_fine_cut_off,
                              kernel1="composite-exponential", kernel2="exponential", 
                              kernel3="exponential", kernel4="exponential",
                              A_coarse_comp=5, A_fine_comp=2,
                              tau_long=10, tau_short=4, 
                              A_coarse=4, A_fine=1,
                              tau=14, 
                              max_weight=15, min_weight=-16,
                              debug=0): 
       
        def clip_newWeight (newWeight, max_weight=max_weight, min_weight=min_weight):
            if newWeight > max_weight:
                newWeight = max_weight
            elif newWeight < min_weight:
                newWeight = min_weight
            return newWeight

        kernel_list = ["composite-exponential", "exponential"]
            
        if successive_correct_cnt >= coarse_fine_cut_off:   # determine A
            A = A_fine
            A_comp = A_fine_comp
            if debug:
                f_handle.write("Instance {}: switching to Fine-update at out-spike time {}\n"
                                .format(instance, spike_out_time))
        else:
            A = A_coarse
            A_comp = A_coarse_comp
        newWeight = [None] * len(oldWeight)
        
        for i in range(len(newWeight)):
            if spike_in_time <= spike_out_time:       # causal 1st or 4th quadrant
                if reward_signal == 1: # causal P+: 1st quadrant
                    if kernel1=="composite-exponential":
                        deltaWeight = \
                            A_comp * (
                                    math.exp(-(spike_out_time - spike_in_time)/tau_long)
                                    - math.exp(-(spike_out_time - spike_in_time)/tau_short)
                                )
                    elif kernel1=="exponential":
                        deltaWeight = \
                            A * (
                                    math.exp(-(spike_out_time - spike_in_time)/tau)
                                )
                    elif not kernel1 in kernel_list:
                        print("Error: kernel1 is not in the kernel list!")
                        exit(1)

                    newWeight[i] = oldWeight[i] + round(deltaWeight)
                    newWeight[i] = clip_newWeight(newWeight[i])

                    if debug and kernel1 in kernel_list:
                        f_handle.write("Instance {}: Causal P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, spike_out_time))
                
                elif reward_signal == 0: # causal P-: 4th quadrant
                    if kernel4=="composite-exponential":
                        deltaWeight = \
                            -A_comp * (
                                    math.exp(-(spike_out_time - spike_in_time)/tau_long)
                                    - math.exp(-(spike_out_time - spike_in_time)/tau_short)
                                )
                    elif kernel4=="exponential":
                        deltaWeight = \
                            -A * (
                                    math.exp(-(spike_out_time - spike_in_time)/tau)
                                )
                    elif not kernel4 in kernel_list:
                        print("Error: kernel4 is not in the kernel list!")
                        exit(1)

                    newWeight[i] = oldWeight[i] + round(deltaWeight)
                    newWeight[i] = clip_newWeight(newWeight[i])
    
                    if debug and kernel4 in kernel_list:
                        f_handle.write("Instance {}: Causal P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, spike_out_time))

            elif spike_in_time > spike_out_time:      # anti-causal 2nd or 3rd quadrant
                if reward_signal == 1: # anti-causal P+: 3rd quadrant
                    if kernel3=="composite-exponential":
                        deltaWeight = \
                            -A_comp * (
                                    math.exp((spike_out_time - spike_in_time)/tau_long)
                                    - math.exp((spike_out_time - spike_in_time)/tau_short)
                                )
                    elif kernel3=="exponential":
                        deltaWeight = \
                            -A * (
                                    math.exp((spike_out_time - spike_in_time)/tau)
                                )

                    elif not kernel3 in kernel_list:
                        print("Error: kernel3 is not in the kernel list!")
                        exit(1)

                    newWeight[i] = oldWeight[i] + round(deltaWeight)
                    newWeight[i] = clip_newWeight(newWeight[i])
        
                    if debug and kernel3 in kernel_list:
                        f_handle.write("Instance {}: anti-Causal P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, spike_out_time))

                elif reward_signal == 0: # anti-causal P-: 2nd quadrant
                    if kernel2=="composite-exponential":
                        deltaWeight = \
                            A_comp * (
                                    math.exp((spike_out_time - spike_in_time)/tau_long)
                                    - math.exp((spike_out_time - spike_in_time)/tau_short)
                                )
                    elif kernel2=="exponential":
                        deltaWeight = \
                            A * (
                                    math.exp((spike_out_time - spike_in_time)/tau)
                                )

                    elif not kernel2 in kernel_list:
                        print("Error: kernel2 is not in the kernel list!")
                        exit(1)

                    newWeight[i] = oldWeight[i] + round(deltaWeight)
                    newWeight[i] = clip_newWeight(newWeight[i])
        
                    if debug and kernel2 in kernel_list:
                        f_handle.write("Instance {}: anti-Causal P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight[i], newWeight[i], causal_fan_in_addr[i], self.neuron_idx, spike_out_time))
    
        
        return newWeight           

    def BRRC_training(self, spike_ref_time, spike_out_time, instance,
                                oldWeight, causal_fan_in_addr, f_handle,
                                reward_signal, isf2f, isIntended,
                                successive_correct_cnt, coarse_fine_cut_off,
                                kernel_f2f_pos="rectangular", kernel_f2f_neg="exponential",
                                kernel_other_pos="exponential", kernel_intended_neg="exponential",
                                A_coarse_comp=5, A_fine_comp=2,
                                tau_long=10, tau_short=4, 
                                A_coarse=2, A_fine=1,
                                tau=9, 
                                t_start=4, t_end=200, A_coarse_rect=1, A_fine_rect=0,
                                max_weight=9, min_weight=-9,
                                debug=0): 

            # isf2f is to indicate whether the neuron being processed is a first-to-spike one
            # isIntended is to indicate whether the neuron being processed is the intended first-to-spike one, only pertinent to nonF2F neurons

            def clip_newWeight (newWeight, max_weight=max_weight, min_weight=min_weight):
                if newWeight > max_weight:
                    newWeight = max_weight
                elif newWeight < min_weight:
                    newWeight = min_weight
                return newWeight

            kernel_list = ["composite-exponential", "exponential", "rectangular"]
                
            if successive_correct_cnt >= coarse_fine_cut_off:   # determine A
                A = A_fine
                A_comp = A_fine_comp
                A_rect = A_fine_rect
                if debug:
                    f_handle.write("Instance {}: switching to Fine-update at out-spike time {}\n"
                                    .format(instance, spike_out_time))
            else:
                A = A_coarse
                A_comp = A_coarse_comp
                A_rect = A_coarse_rect
            newWeight = oldWeight
            
            # weight update for f2f neuron, t_ref=t_in 
            if isf2f:
                # F2F P+ reward quadrant
                if reward_signal:
                    if kernel_f2f_pos=="composite-exponential":
                        deltaWeight = \
                            A_comp * (
                                    math.exp(-(spike_out_time - spike_ref_time)/tau_long)
                                    - math.exp(-(spike_out_time - spike_ref_time)/tau_short)
                                )
                    elif kernel_f2f_pos=="exponential":
                        deltaWeight = \
                            A * (
                                    math.exp(-(spike_out_time - spike_ref_time)/tau)
                                )
                    elif kernel_f2f_pos == "rectangular":
                        if ((spike_out_time - spike_ref_time)<=t_end) \
                            and ((spike_out_time - spike_ref_time)>=t_start):
                            deltaWeight = A_rect
                        else:
                            deltaWeight = 0
                    elif not kernel_f2f_pos in kernel_list:
                        print("Error when calling BRRC_training: kernel_f2f_pos is not in the kernel list!")
                        exit(1)

                    newWeight = oldWeight + round(deltaWeight)
                    newWeight = clip_newWeight(newWeight)
                    if debug:
                        f_handle.write("Instance {}: F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight, newWeight, causal_fan_in_addr, self.neuron_idx, spike_out_time))                           
                # F2F P- punishment quadrant
                else:
                    if kernel_f2f_neg=="composite-exponential":
                        deltaWeight = \
                            -A_comp * (
                                    math.exp(-(spike_out_time - spike_ref_time)/tau_long)
                                    - math.exp(-(spike_out_time - spike_ref_time)/tau_short)
                                )
                    elif kernel_f2f_neg=="exponential":
                        deltaWeight = \
                            -A * (
                                    math.exp(-(spike_out_time - spike_ref_time)/tau)
                                )
                    elif not kernel_f2f_neg in kernel_list:
                        print("Error when calling BRRC_training: kernel_f2f_neg is not in the kernel list!")
                        exit(1)

                    newWeight = oldWeight + round(deltaWeight)
                    newWeight = clip_newWeight(newWeight)

                    if debug:
                        f_handle.write("Instance {}: F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight, newWeight, causal_fan_in_addr, self.neuron_idx, spike_out_time))                           

            # weight update for non-f2f neuron that fired within separation window, t_ref = t_first-spike
            else:
                # weight update for non-f2f neurons that fired within separation window under reward
                if reward_signal and not isIntended:
                    if kernel_other_pos=="composite-exponential":
                        deltaWeight = \
                            -A_comp * (
                                    math.exp(-(spike_out_time - spike_ref_time)/tau_long)
                                    - math.exp(-(spike_out_time - spike_ref_time)/tau_short)
                                )
                    elif kernel_other_pos=="exponential":
                        deltaWeight = \
                            -A * (
                                    math.exp(-(spike_out_time - spike_ref_time)/tau)
                                )
                    elif not kernel_other_pos in kernel_list:
                        print("Error when calling BRRC_training: kernel_other_pos is not in the kernel list!")
                        exit(1)

                    newWeight = oldWeight + round(deltaWeight)
                    newWeight = clip_newWeight(newWeight)

                    if debug:
                        f_handle.write("Instance {}: Non-F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight, newWeight, causal_fan_in_addr, self.neuron_idx, spike_out_time))                           
                
                # weight update for non-f2f neuron but is intended that fired within separation window under punishment
                elif not reward_signal and isIntended:
                    # if non-F2F intended neruon has fired
                    if isinstance(spike_out_time, int):
                        if kernel_intended_neg=="composite-exponential":
                            deltaWeight = \
                                A_comp * (
                                        math.exp(-(spike_out_time - spike_ref_time)/tau_long)
                                        - math.exp(-(spike_out_time - spike_ref_time)/tau_short)
                                    )
                        elif kernel_intended_neg=="exponential":
                            deltaWeight = \
                                A * (
                                        math.exp(-(spike_out_time - spike_ref_time)/tau)
                                    )
                        elif not kernel_intended_neg in kernel_list:
                            print("Error when calling BRRC_training: kernel_intended_neg is not in the kernel list!")
                            exit(1)
                    # if non-F2F intended neuron has not fired
                    else:
                        deltaWeight = 1
                    
                    newWeight = oldWeight + round(deltaWeight)
                    newWeight = clip_newWeight(newWeight)
                    if debug:
                        f_handle.write("Instance {}: Non-F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on Neuron {} upon out-spike at time {}\n"
                        .format(instance, oldWeight, newWeight, causal_fan_in_addr, self.neuron_idx, spike_out_time))                           

            return newWeight           


    def accumulate(self, sim_point, spike_in_info, WeightRAM_inst, instance, f_handle,
                   debug_mode=0
                   ):     
        # spike_in_info is the data transmitted between neurons: (a dictionary)
        #   spike_in_info["fired_synapse_addr"] (a list of int)
        #   spike_in_info["time"] (an int)
        #   simpoint is in range(0,duration,dt)         
        
        dt = SpikingNeuron.dt

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
                for i in range(len(relavent_fan_in_addr)):
                    self.spike_in_cache.writeSpikeInInfo(
                                        fired_synapse_addr=relavent_fan_in_addr[i],
                                        time=sim_point,
                                        weight = weight[i]   
                                        )

                if self.fire_cnt == -1:
                # only update causal_spike_in_info when the neuron has not fired
                    # choose the latest in-spike to be considered as the causal one
                    self.oldWeight = weight[-1]
                    self.causal_spike_in_info = self.last_spike_in_info
                    self.causal_fan_in_addr = relavent_fan_in_addr[-1]
                    
                self.u[sim_point] = (1 - dt/self.tau_u) * self.u[sim_point-1] + sum(weight) * dt
            else:
                if (self.u[sim_point-1] != 0):                          # added to take advantage of potential sparsity
                    self.u[sim_point] = (1 - dt/self.tau_u) * self.u[sim_point-1]

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
                    f_handle.write("Instance {}: Neuron {} at Layer{} has fired {} times at step {}\n"
                        .format(instance, self.neuron_idx, self.layer_idx,self.fire_cnt + 1, [entry["time"] for entry in self.spike_out_info]))               
