import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import NetworkConnectivity

#%%

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
    
def getInLatencies(in_pattern, early_latency_list, late_latency_list,
                    mean_early, std_early, mean_late, std_late, low_lim=0, high_lim=64,
                    num_edge_maps=4, num_blocks=16):

    # num_in_neurons = num_edge_maps * num_blocks
    in_pattern_list = ["9", "8", "6", "3"]
    num_in_neurons = num_edge_maps * num_blocks
    if not in_pattern in in_pattern_list:
        print("Error when calling getInLatencies: illegal specification of \"in_pattern\"")
        exit(1)
    
    # InLatencies[edge_map_idx][block_idx]
    InLatencies = [[None]*num_blocks] * num_edge_maps
    for map_idx in range(num_edge_maps):
        for block_idx in range(num_blocks):
            latency_late = \
                BimodalLatency("late", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[map_idx][block_idx] = latency_late
            late_latency_list.append(latency_late)
    
    if in_pattern == "9":
        # feature map 0: '--'
        for block_idx in [1, 2]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [1, 10, 13]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in [5, 6, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [2, 5, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 
            
    elif in_pattern == "6":
        # feature map 0: '--'
        for block_idx in [13, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [2, 5, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in [5, 9, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [5, 10, 13]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 

    elif in_pattern == "8":
        # feature map 0: '--'
        for block_idx in [1, 2, 13, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [6, 9]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in []:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [5, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 

    elif in_pattern == "3":
        # feature map 0: '--'
        for block_idx in [1, 2, 13, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[0][block_idx] = latency_early 
        # feature map 1: '/'
        for block_idx in [6, 14]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[1][block_idx] = latency_early 
        # feature map 2: '|'
        for block_idx in []:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[2][block_idx] = latency_early 
        # feature map 3: '\'
        for block_idx in [2, 10]:
            latency_early = \
                BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
            InLatencies[3][block_idx] = latency_early 
    

    return (InLatencies, early_latency_list, late_latency_list)

def plotInLatencyDistribution(early_latency_list, late_latency_list, tau_u, num_bins=8):
    fig, ax = plt.subplots(figsize=(14,7))
    early_list_tau = [latency / tau_u for latency in early_latency_list]
    late_list_tau = [latency / tau_u for latency in late_latency_list]
    ax.hist([early_list_tau, late_list_tau], bins=num_bins, label=['early latency','late latency'],
            edgecolor='k')

    ax.set_title('Stimulus Spike Latency Distribution in units of ' + r'$\tau_u$'
                ,fontsize=18, fontweight='bold')
    ax.axvline(x=num_bins/2, linewidth=2, color='k', linestyle='dashed')
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(num_bins/2, max_ylim*0.8, r'early-late cutoff ${}\tau_u$'.format(int(num_bins/2)),
            fontsize=12)
    ax.set_xticks(range(0, num_bins+1))
    xticklabel_list = [r'{}$\tau_u$'.format(i) for i in range(0, num_bins+1)]
    ax.set_xticklabels(xticklabel_list)
    ax.legend(fontsize=15)


#%% Parameters to tune
######################################################################################
printout_dir = "sim_printouts/FourEdgeMaps4Classes/"
num_categories = 4
num_edge_maps = 4
W_input = 4
F_hidden = 3
S_hidden = 1
depth_hidden_per_sublocation = 5


input_connectivity, hidden_connectivity, output_connectivity = \
    NetworkConnectivity.initializeNetWorkConnectivity(
        num_categories=num_categories, num_edge_maps=num_edge_maps, W_input=W_input,
        F_hidden=F_hidden, S_hidden=S_hidden, 
        depth_hidden_per_sublocation=depth_hidden_per_sublocation
    )