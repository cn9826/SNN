import numpy as np
import matplotlib.pyplot as plt
import SpikingNeuron
import math
#%% Single Neuron Dynamics
sn0 = SpikingNeuron.SpikingNeuron(layer_idx=0, neuron_idx=0, tau_u=16, tau_v=100, threshold=500, t_refract=50)
# spike_time=[30, 100, 200, 300, 340]
# weight = [3, 10, -5, 15, 15]
spike_time = [100, 200]
weight = [10, 10]
sn0.accumulateOnSpike(spike_time=spike_time, weight=weight)
t = np.arange(0,sn0.num_steps,1) 

## evaluate u(t) using formula
u_formula = (weight[0] * np.exp(-(t-spike_time[0])/sn0.tau_u) * SpikingNeuron.heaviside(t-spike_time[0]) 
                        +  weight[1] * np.exp(-(t-spike_time[1])/sn0.tau_u) * SpikingNeuron.heaviside(t-spike_time[1])) 

v_formula = sn0.tau_u*(weight[0] * (1-np.exp(-(t-spike_time[0])/sn0.tau_u)) * SpikingNeuron.heaviside(t-spike_time[0]) 
                        + weight[1] * (1-np.exp(-(t-spike_time[1])/sn0.tau_u)) * SpikingNeuron.heaviside(t-spike_time[1])) 


fig1= plt.figure()
fig1.subplots_adjust(hspace=0.5)

hax1 = plt.subplot(3, 1, 1)
x_pos = spike_time
y_pos = [0] * len(spike_time)
x_direct = [0] * len(spike_time)
y_direct = [1] * len(spike_time)
plt.xlim((t[0], t[-1]))
plt.ylim((0,2))
hax1.quiver(x_pos, y_pos, x_direct, y_direct, scale=10 ,color='C0', width=0.005)
hax1.set(title="Input Spiking Event")
hax1.get_yaxis().set_ticks([])

hax2 = plt.subplot(3, 1, 2)
plt.plot(t, sn0.u, '-', lw=2)
plt.plot(t, u_formula, '-.', color='r', lw=2)

plt.xlim((t[0], t[-1]))
plt.grid(True)
hax2.set(title="Synaptic Current " + r"$u(t)$")


hax3 = plt.subplot(3, 1, 3)
plt.plot(t, sn0.v, '-', lw=2)
plt.plot(t, v_formula, '-.', color='r', lw=2)
hax3.set(
            title="Membrane Potential " + r"$v(t)$",
            xlabel="Timestep"
        )
hax3.hlines(y=sn0.threshold, xmin=t[0], xmax=t[-1], lw=2, color='0.2', linestyles='dashed')
# plt.text(400, 3.15, "Threshold", fontsize=12)
hax3.vlines(x=[sn0.refractory_window[0], sn0.refractory_window[-1]], ymin=0, ymax=4, lw=2, color='r', linestyles='dotted')
plt.xlim((t[0], t[-1]))
# plt.ylim((0,4))
plt.grid(True)
fig1.show()


plt.show()
