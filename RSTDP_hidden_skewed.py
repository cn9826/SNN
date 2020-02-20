def RSTDP_hidden_skewed(self, spike_in_time, spike_out_time, instance,
                    oldWeight, fan_in_addr, neuron_causal_tag, f_handle, 
                    reward_signal, isf2f, isIntended, 
                    successive_correct_cnt, coarse_fine_cut_off,
                    A_pos_coarse=2, A_pos_fine=1, tau_pos=30,
                    A_neg_coarse=1, A_neg_fine=1, tau_neg=30,
                    t_start=0, 
                    max_weight=7, min_weight=0,
                    debug=0
                    ):
        if not isIntended:
            print("Error when calling SpikingNeuron.RSTDP_hidden(): isIntended {} is not 1 for Neuron {}"
                .format(isIntended, self.neuron_idx))
            exit(1)

        if len(fan_in_addr) != len(oldWeight):
            print("Error when calling SpikingNeuron.RSTDP_hidden(): len(fan_in_addr) {} does not correspond to len(oldWeight) {}!"
                .format(len(fan_in_addr), len(oldWeight))) 
            exit(1)
        if len(spike_in_time) != len(oldWeight):
            print("Error when calling SpikingNeuron.RSTDP_hidden(): len(spike_in_time) {} does not correspond to len(oldWeight) {}!"
                .format(len(spike_in_time), len(oldWeight))) 
            exit(1)

        if neuron_causal_tag:
            neuron_causal_str = "Causal       "
        elif not neuron_causal_tag:
            neuron_causal_str = "anti-Causal  "

        if successive_correct_cnt >= coarse_fine_cut_off:
            A_pos = A_pos_fine
            A_neg = A_neg_fine
        else:
            A_pos = A_pos_coarse
            A_neg = A_neg_coarse
        
        newWeight = [None for i in range(len(oldWeight))]

        for i in range(len(oldWeight)):
            s = spike_out_time - spike_in_time[i] 
            # if updating a causal presynaptic hidden neuron
            if neuron_causal_tag:
                # causal synaptic weights update on a causal presynaptic hidden neuron
                if s >= 0:
                    if s <= t_start:
                        deltaWeight = 0
                    else:
                        deltaWeight = round(A_pos * math.exp(-s/tau_pos))
                # anti-causal synaptic weights update on a causal presynaptic hidden neuron
                elif s < 0:
                    if (-s) <= t_start:
                        deltaWeight = 0
                    else:
                        deltaWeight = -1 * round(A_neg * math.exp(s/tau_neg))
                newWeight[i] = oldWeight[i] + deltaWeight
                newWeight[i] = clip_newWeight(newWeight=newWeight[i], max_weight=max_weight, min_weight=min_weight)
            
            # if updating an anti-causal presynaptic hidden neuron            
            elif not neuron_causal_tag:
                # causal synaptic weights update on an anti-causal presynaptic hidden neuron
                if s >= 0:
                    if s <= t_start:
                        deltaWeight = 0
                    else:
                        deltaWeight = -1 * round(A_neg * math.exp(-s/tau_neg))
                # anti-causal synaptic weights update on an anti-causal presynaptic hidden neuron
                elif s < 0:
                    if (-s) <= t_start:
                        deltaWeight = 0
                    else:
                        deltaWeight = round(A_pos * math.exp(s/tau_pos))
                newWeight[i] = oldWeight[i] + deltaWeight
                newWeight[i] = clip_newWeight(newWeight=newWeight[i], max_weight=max_weight, min_weight=min_weight)
                
            if debug:
                if reward_signal and isf2f:
                    f_handle.write("Instance {}: F2F P+ update oldWeight: {} to newWeight: {} of Synapse {} on {} Neuron {} upon out-spike at time {}\n"
                    .format(instance, oldWeight[i], newWeight[i], fan_in_addr[i], neuron_causal_str, self.neuron_idx, spike_out_time))                           
                elif (not reward_signal) and (not isf2f):
                    f_handle.write("Instance {}: Non-F2F P- update oldWeight: {} to newWeight: {} of Synapse {} on {} Neuron {} upon out-spike at time {}\n"
                    .format(instance, oldWeight[i], newWeight[i], fan_in_addr[i], neuron_causal_str, self.neuron_idx, spike_out_time))                           
                
        return newWeight