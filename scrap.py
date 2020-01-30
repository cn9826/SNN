from operator import itemgetter
import numpy as np
import random
import math
import matplotlib.pyplot as plt



#%% numpy stuff
## indexing multidimensional arrays 
# y = np.arange(35).reshape(5,7)
# print(np.array([0, 2, 4]))
# print("y = \n {}\n".format(y))
# print("index array =\n {}\n".format([np.array([0,2,4]), np.array([0,1,2])]))
# print(y[np.array([0,2,4]), np.array([0,1,2])])
# print(y[1, np.array([0,2,4])])
# print("y[(1, 1)] =\n {}\n".format(y[(1,1)]))
# ## Boolean index arrays: result will be multidimensional if y has more dimensions than b
# b = y>20
# print("b = \n{}\n".format(b))
# print("y[b] = \n{}\n".format(y[b]))
# print("b[:,5] = \n{}\n".format(b[:,5]))
# print("y[b[:,5]] = \n{}\n".format(y[b[:,5]]))

# ## Combining index arrays with slices
# print("y[np.array([0,2,4]), 1:3] = \n {}\n".format(y[np.array([0,2,4]), 1:3]))

# ## Structural indexing tools: np.newaxis object used to add new dimensions with a size of 1
# x = np.arange(5)
# print("x = \n{}\n".format(x))
# print("x[:, np.newaxis] = \n{}\n".format(x[:, np.newaxis]))
# print("x[np.newaxis, :] = \n{}\n".format(x[np.newaxis, :]))

# ## Assigning values to indexed arrays:
# # unlike sliceing references, assignments are always made to the original data in the array
# x = np.arange(0, 50, 10)
# x[np.array([1, 1, 3, 1])] += 1
# print("x = \n{}\n".format(x))

# ## Dealing with variable number of indices with programs:
# # supply a tuple to the index and the tuple will be interpreted asa  list of indicies
# z = np.arange(81).reshape(3,3,3,3)
# print("z = \n{}\n".format(z))

# indices = (1, 1, 1, slice(0,2))
# #%% Parameters to tune
# ######################################################################################

# ## Specify Global Connectivity Parmeters
# num_neurons_perLayer = [8,8]                           # Assuming num_neurons_perLayer is the number of connections in FC case
# max_num_fires = 1

# ## Specify common Spiking Neuron Parameters
# duration = 80
# tau_u = 8      # in units with respect to duration
# tau_v = None     # in units with respect to duration
# vth_low = 1
# vth_high = 140

# ## Supervised Training Parameters
# supervised_training_on = 1      # turn on/off supervised training 
# separation_window = 12
# stop_num = 150
# coarse_fine_ratio=0.4

# ## Training Dataset Parameters
# num_instances =5000              # number of training instances per epoch

# ## Simulation Settings
# debug_mode = 1
# plot_response = 0
# plot_InLatency_dist = 1



# ######################################################################################
# #%%
# def BimodalLatency(latency_mode, mean_early, std_early, mean_late, std_late, low_lim=0, high_lim=64):
#     latency_mode_list = ["early", "late"]
#     if not latency_mode in latency_mode_list:
#         print("Error when calling BimodalLatency: illegal specification of \"latency\"")
#         exit(1)
#     if latency_mode == "early":
#         latency = np.random.normal(mean_early, std_early)
#         if latency < low_lim:
#             latency = low_lim
#     elif latency_mode == "late":
#         latency = np.random.normal(mean_late, std_late)
#         if latency > high_lim:
#             latency = high_lim
#     return int(latency)
    
# def getInLatencies(in_pattern, num_in_neurons, early_latency_list, late_latency_list,
#                     mean_early, std_early, mean_late, std_late, low_lim=0, high_lim=64):

#     in_pattern_list = ["O", "X", "<<", "//", ">>", "UA", "DA", "BS"]
#     if not in_pattern in in_pattern_list:
#         print("Error when calling getInLatencies: illegal specification of \"in_pattern\"")
#         exit(1)
    
#     InLatencies = [None] * num_in_neurons
#     for i in range(num_in_neurons):
#         latency_late = \
#             BimodalLatency("late", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#         InLatencies[i] = latency_late
#         late_latency_list.append(latency_late)

#     if in_pattern == "O":
#         for i in [0, 3, 5, 6]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == "X":
#         for i in [1, 2, 4, 7]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == "<<":
#         for i in [0, 1, 6, 7]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == "//":
#         for i in [0, 1, 2, 3]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == ">>":
#         for i in [2, 3, 4, 5]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == "UA":
#         for i in [0, 2, 5, 7]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == "DA":
#         for i in [1, 3, 4, 6]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     elif in_pattern == "BS":
#         for i in [4, 5, 6, 7]:
#             latency_early = \
#                 BimodalLatency("early", mean_early, std_early, mean_late, std_late, low_lim, high_lim)
#             InLatencies[i] = latency_early
#             early_latency_list.append(latency_early)

#     return (InLatencies, early_latency_list, late_latency_list)

# def plotInLatencyDistribution(early_latency_list, late_latency_list, tau_u, num_bins=8):
#     fig, ax = plt.subplots(figsize=(14,7))
#     early_list_tau = [latency / tau_u for latency in early_latency_list]
#     late_list_tau = [latency / tau_u for latency in late_latency_list]
#     ax.hist([early_list_tau, late_list_tau], bins=num_bins, label=['early latency','late latency'],
#             edgecolor='k')

#     ax.set_title('Stimulus Spike Latency Distribution in units of ' + r'$\tau_u$'
#                 ,fontsize=18, fontweight='bold')
#     ax.axvline(x=num_bins/2, linewidth=2, color='k', linestyle='dashed')
#     min_ylim, max_ylim = ax.get_ylim()
#     ax.text(num_bins/2, max_ylim*0.8, r'early-late cutoff ${}\tau_u$'.format(int(num_bins/2)),
#             fontsize=12)
#     ax.set_xticks(range(0, num_bins+1))
#     xticklabel_list = [r'{}$\tau_u$'.format(i) for i in range(0, num_bins+1)]
#     ax.set_xticklabels(xticklabel_list)
#     ax.legend(fontsize=15)

# ## Define Input & Output Patterns
# mean_early = 0*2*tau_u + 3*tau_u
# std_early = int(3*tau_u/3)
# mean_late = 4*2*tau_u - 3*tau_u
# std_late = int(3*tau_u/3)

# initial_weight = [6] * num_neurons_perLayer[-2] * num_neurons_perLayer[-1] 
# weight_vector = \
#     [
#         10, 10, 10, 10, 10, 10, 10, 10,
#         *initial_weight
#     ]

# input_patterns = ("O", "X", "<<", "//", ">>", "UA", "DA", "BS")

# output_pattern = \
#     {
#         "O"      :   sum(num_neurons_perLayer[0:-1]),
#         "X"      :   sum(num_neurons_perLayer[0:-1]) + 1,
#         "<<"     :   sum(num_neurons_perLayer[0:-1]) + 2,
#         "//"     :   sum(num_neurons_perLayer[0:-1]) + 3,
#         ">>"     :   sum(num_neurons_perLayer[0:-1]) + 4,
#         "UA"     :   sum(num_neurons_perLayer[0:-1]) + 5,
#         "DA"     :   sum(num_neurons_perLayer[0:-1]) + 6,
#         "BS"     :   sum(num_neurons_perLayer[0:-1]) + 7
#     }

# ## Create stimulus spikes at the inuput layer (layer 0)
# # Dimension: num_instances x num_input_neurons
# stimulus_time_vector = [
#                             {
#                                 "in_pattern"    :    None,
#                                 "in_latency"    :    [None] * num_neurons_perLayer[0]
#                             }
#                             for instance in range(num_instances)
#                        ]

# early_latency_list = []
# late_latency_list = []
# for instance in range(num_instances):
#     stimulus_time_vector[instance]["in_pattern"] = \
#             random.choice(input_patterns)
#     stimulus_time_vector[instance]["in_latency"], early_latency_list, late_latency_list = \
#             getInLatencies(in_pattern=stimulus_time_vector[instance]["in_pattern"],
#                            num_in_neurons=num_neurons_perLayer[0],
#                            early_latency_list = early_latency_list,
#                            late_latency_list = late_latency_list,
#                            mean_early=mean_early, std_early=std_early,
#                            mean_late=mean_late, std_late=std_late,
#                            low_lim=0, high_lim=mean_late+2*tau_u  
#                            )

# if plot_InLatency_dist:
#     plotInLatencyDistribution(early_latency_list, late_latency_list, tau_u, num_bins=8)
#     plt.show()


#%%
# BS_str = r"\\"
# if BS_str == r"\\":
#     print(BS_str)

# num_neurons_perLayer = [8, 24, 10]       # Assuming num_neurons_perLayer is the number of connections in FC case
# num_connect_perNeuron = [1,4,-1]        # -1 denotes FC       

# num_in_spikes_hidden = 2
# num_in_spikes_output = 4

# max_num_fires = 1


# fan_in_neuron = [
#                     [], [], [], [], [], [], [], [],
#                     *[[0, 1, 4, 5]] * int(num_neurons_perLayer[1]/2),
#                     *[[2, 3, 6, 7]] * int(num_neurons_perLayer[1]/2),
#                     *[[x for x in range(8, 32)]] * num_neurons_perLayer[2]  
#                 ]
# print(fan_in_neuron)

#%%
# A_pos = 1 
# tau_pos = 10
# for s in range(-30, 1, 1):
#     deltaWeight = round(A_pos * math.exp(s/tau_pos))
#     print("s = {}\t\t deltaWeight = {}".format(s, deltaWeight))

#%%
list_2d = \
    [
        [1, 2, 3], [4, 5, 6], [7, 8, 9]
    ]
list_1d = [ x for sub_list in list_2d for x in sub_list]
print(list_1d[0:-1])

lst1 = [2, None, 3, None]
