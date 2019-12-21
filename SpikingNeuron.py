import numpy as np

def heaviside(t):
    unit_step = np.arange(t.shape[0])
    lcv = np.arange(t.shape[0])
    for place in lcv:
        if t[place] == 0:
            unit_step[place] = 1
        elif t[place] > 0:
            unit_step[place] = 1
        elif t[place] < 0:
            unit_step[place] = 0
    return unit_step

class SpikingNeuron:
    # shared Class Variables
    num_steps = 500                 # number of time steps for entire simulation
    dt = 1

    
    # Specific Object Variables
    def __init__(self, layer_idx, neuron_idx, tau_u, tau_v, threshold, t_refract):       # Later add num_connections, preNeuron_idx, synaptic_weights etc...
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.u = [0] * SpikingNeuron.num_steps          
        self.v = [0] * SpikingNeuron.num_steps
        self.tau_u = tau_u                  # current decay constant, in units of time step       
        self.tau_v = tau_v                  # potential decay constant, in units of time step
        self.threshold = threshold          # potential threshold
        self.t_refract = t_refract          # refractory time after spike, in units of time step
        self.refractory_window = [0] * t_refract
        self.spike_out_stamp = None         # output spike time stamp


    def accumulateOnSpike(self, spike_time, weight):        # spike_time & weight are lists that each corresponds to a firing pre-synaptic neuron   
        dt = SpikingNeuron.dt
        if (len(spike_time) != len(weight)):
            print("The number of elements in list 'spike_time' does not match the number of elements in list 'weight'")
            exit(1)
        
        spike_idx = 0                                       # temporary, Later...
        
        for i in range(1,SpikingNeuron.num_steps):
            # update synaptic current 
            if (i in spike_time):
                self.u[i] = (1 - dt/self.tau_u) * self.u[i-1] + dt * weight[spike_idx]
                spike_idx = spike_idx + 1
            else:
                self.u[i] = (1 - dt/self.tau_u) * self.u[i-1]
            
            # update membrane potential
                # check if neuron has spiked: if spiked, form a refractory time window where update is disabled
            if (self.v[i-1] >= self.threshold):
                # form a timestep window when neuron is in refractory period
                self.refractory_window  = np.arange(i, i + self.t_refract, 1)
                
                self.v[self.refractory_window[0]:self.refractory_window[-1]+1] = [0] * len(self.refractory_window)
            
            if ((i not in self.refractory_window) or (i == 0)):
                if self.tau_v is not None:
                    self.v[i] = (1-dt/self.tau_v) * self.v[i-1] + self.u[i]*dt
                else:
                    self.v[i] = self.v[i-1] + self.u[i]*dt
        

