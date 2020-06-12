import SNN
import numpy as np
import math
import matplotlib.pyplot as plt
import NetworkConnectivity
import codecs
import json
from datetime import datetime
from pathlib import Path
import shutil

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


def createMovingAccuracyFigure(num_instances):
    fig, ax = plt.subplots(figsize=(14, 7))
    xticklabel_list = ['{}'.format(i) for i in range(0, num_instances+1, 1000)]

    ax.set_xlim(0, num_instances + 1)
    ax.set_xticks(range(0, num_instances+1, 1000))
    ax.set_xticklabels(xticklabel_list)
    #ax.set_xticklabels(xticklabel_list, fontsize=6, fontWeight='bold')
    ax.set_xlabel('Number of Instances', fontsize=14, fontweight='bold')

    yticklabel_list = ['{0:3.2f}'.format(i) for i in np.arange(0, 1.1, 0.05)]
    ax.set_yticks(np.arange(0, 1.1, 0.05))
    ax.set_yticklabels(yticklabel_list)
    ax.set_ylabel('Moving Accuracy During Training', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)

    ax.grid(which='both', axis='y')
    return (fig, ax)


def ItLmapping(pooled_array, window, kernel, alpha=0.5, poly_degree=1):
    # the intensity-to-latency mapping function
    # mapping kernels are "exponential" and "polynomial"

    # pooled_array is a ndnumpy array with shape(# of instances, W, W, 4) whose values
    # have been normalized to 1
    # kernel is one of the: "exponential" or "polynomial"
    # window is the encoding window
    # alpha is the scaling factor in front of the exponential term
    # poly_degree is the degree of polynomial should "polynomial" be chosen as the mapping kernel

    kernel_lst = ["exponential", "polynomial"]

    if kernel not in kernel_lst:
        print("Error when calling ItLmapping: specified kernel {} is not supported!"
              .format(kernel))
        print("Defaulting kernel to be \"exponential\"!")
        kernel = "exponential"

    if kernel == "exponential":
        beta = math.log(( window + 1) / alpha)
        LatencyArray = \
            np.rint(-1 * alpha * np.exp(beta*pooled_array) + window + 1)
    elif kernel == "polynomial":
        LatencyArray = \
            np.rint(window * (-1 * np.power(pooled_array, poly_degree) + 1))

    return LatencyArray


def hist_latency(latency_array, tau_u, num_bins=8):
    latency_array_tau = latency_array / tau_u
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xticks(range(0, num_bins + 1))
    xticklabel_list = [r'{}$\tau_u$'.format(i) for i in range(0, num_bins + 1)]
    ax.set_xticklabels(xticklabel_list)
    ax.hist(latency_array_tau.flatten(), bins=num_bins, edgecolor='k')
    ax.set_ylim(0,1e6)
    ax.set_title('Stimulus Spike Latency Distribution -- filtered and pooled MNIST',
                 fontsize=18, fontweight='bold')


def getTrainingAccuracy(moving_window, plot_on=0):
    if None in moving_window:
        return 0
    else:
        num_instances = len(moving_window)
        correct_total = sum(moving_window)
        accuracy = correct_total / num_instances
        return accuracy

def getUntrainedPerc(instance, WeightRAM_inst, W_hidden, num_categories,
                     num_input_neurons, num_fan_in_hidden,
                     num_fan_in_output, appendUntrainedStats, f_handle_training_stats,
                     ax_accuracy
                     ):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f_handle_training_stats.write("--------------Instance {}------------------------------\n"
                                  .format(instance))
    f_handle_training_stats.write("{}\n".format(dt_string))

    total_trained_perc = \
        sum(WeightRAM.dirty[num_input_neurons:]) \
        / (len(WeightRAM.dirty) - num_input_neurons)
    print(
        "Total untrained Synapse Percentage:{perc:.2%}\n"
            .format(perc=1 - total_trained_perc)
    )
    f_handle_training_stats.write(
        "Total untrained Synapse Percentage:{perc:.2%}\n\n"
            .format(perc=1 - total_trained_perc)
    )

    hidden_untrained_bins = [0] * W_hidden ** 2
    hidden_trained = \
        sum(WeightRAM.dirty[num_input_neurons:num_input_neurons + num_fan_in_hidden])
    hidden_trained_perc = hidden_trained / num_fan_in_hidden

    hidden_untrained = num_fan_in_hidden - hidden_trained

    for synapse_idx in range(num_input_neurons, num_input_neurons + num_fan_in_hidden):
        if WeightRAM.dirty[synapse_idx] == 0:
            sublocation_idx = WeightRAM.post_neuron_location[synapse_idx]
            hidden_untrained_bins[sublocation_idx] += 1

    f_handle_training_stats.write(
        "Untrained Fan-in Synapse in Hidden layer:{perc:.2%}\n"
            .format(perc=1 - hidden_trained_perc)
    )

    f_handle_training_stats.write(
        "Untrained Fan-in Synapses breakdown by sublocation:\n"
    )
    for sublocation_idx in range(W_hidden ** 2):
        f_handle_training_stats.write(
            "location_idx: {0:2d}\t\tUntrained Percentage: {1:.2%}\n"
                .format(sublocation_idx, hidden_untrained_bins[sublocation_idx] / num_fan_in_hidden)
        )

    f_handle_training_stats.write("\n")

    output_untrained_bins = [0] * num_categories
    output_trained_perc = \
        sum(WeightRAM.dirty[num_input_neurons + num_fan_in_hidden:
                            num_input_neurons + num_fan_in_hidden + num_fan_in_output]) \
        / num_fan_in_output
    for synapse_idx \
            in range(num_input_neurons + num_fan_in_hidden,
                     num_input_neurons + num_fan_in_hidden + num_fan_in_output):
        if WeightRAM.dirty[synapse_idx] == 0:
            category_idx = WeightRAM.post_neuron_location[synapse_idx]
            output_untrained_bins[category_idx] += 1
    f_handle_training_stats.write(
        "Untrained Fan-in Synapse in Output layer:{perc:.2%}\n"
            .format(perc=1 - output_trained_perc)
    )
    f_handle_training_stats.write(
        "Untrained Fan-in Synapses breakdown by label:\n"
    )
    for category_idx in range(num_categories):
        f_handle_training_stats.write(
            "location_idx: {0:2d}\t\tUntrained Percentage: {1:.2%}\n"
                .format(category_idx, output_untrained_bins[category_idx] / num_fan_in_output)
        )

    f_handle_training_stats.write("----------------------------------------------------\n\n")
    if appendUntrainedStats:
        appendUntrainedPerc(ax_accuracy, instance, 1 - total_trained_perc)



def appendAccuracy(ax, instance, accuracy, marker_size=6):
    ax.scatter(instance, accuracy, marker='o', color='r', s=marker_size)
    plt.pause(0.0001)

def appendUntrainedPerc(ax, instance, untrained_perc, marker_size=6):
    ax.scatter(instance, untrained_perc, marker='o', color='g', s=marker_size)
    ax.text(instance, untrained_perc, "{0:.2%}".format(untrained_perc), fontsize=8)
    plt.pause(0.0001)

def imshow_pooled(pooled_arr, num_edge_maps=4):
    # pooled_arr is a 3D array of size (W_hidden, W_hidden, 4)
    fig, ax = plt.subplots(nrows=1, ncols=num_edge_maps, figsize=(15,8))
    fig.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0)
    fig.tight_layout()

    for edge_idx in range(num_edge_maps):
        ax[edge_idx].imshow(pooled_arr[:, :, edge_idx], cmap='gray')

def saveWeightRAM(instance, WeightRAM_inst, identifier):
    ## expect identifier to be a string like "Fig_37"
    save_path = "./WeightRAM/" + identifier + "/"

    Path(save_path).mkdir(parents=True, exist_ok=True)
    json.dump(
        list(WeightRAM_inst.synapse_addr),
        codecs.open(save_path + "synapse_idx_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
    json.dump(
        list(WeightRAM_inst.weight),
        codecs.open(save_path + "weight_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
    json.dump(
        list(WeightRAM_inst.pre_neuron_idx),
        codecs.open(save_path + "pre_neuron_idx_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
    json.dump(
        list(WeightRAM_inst.post_neuron_idx),
        codecs.open(save_path + "post_neuron_idx_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
    json.dump(
        list(WeightRAM_inst.dirty),
        codecs.open(save_path + "dirty_idx_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
    json.dump(
        list(WeightRAM_inst.post_neuron_layer),
        codecs.open(save_path + "post_neuron_layer_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
    json.dump(
        list(WeightRAM_inst.post_neuron_location),
        codecs.open(save_path + "post_neuron_location_" + str(instance) + ".json", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )
# %% define SNN parameters
################################################################
## Specify dataset parameters
num_categories = 10
num_edge_maps = 4

## Specify hidden layer filter size
W_input = 8
F_hidden = 3
S_hidden = 1
depth_hidden_per_sublocation = 10

## Specify common Spiking Neuron Parameters
duration = 80
tau_u = 8
tau_v = None
vth_input = 1
vth_hidden = 350    # with 9-spike consideration: [(9-1) x 5 x tau_u, *9 x 5 x tau_u)
                    # with 9-spike consideration: [(9-1) x 7 x tau_u, 9 x 7 x tau_u)

vth_output = 672   # with 12-spike consideration: [(12-1) x 5 x tau_u, 12 x 5 x tau_u)
                    # with 12-spike consideration: [(12-1) x 7 x tau_u, *12 x 7 x tau_u)

## Supervised Training Parameters
supervised_hidden = 1      # turn on/off supervised training in hidden layer
supervised_output = 1      # turn on/off supervised training in output layer
separation_window = 10
stop_num = 500

accuracy_th = 0.85           # the coarse/fine cutoff for weight update based on moving accuracy
size_moving_window = 1000    # the size of moving window that dynamically calculates inference accuracy during training

## Training Dataset Parameters
num_instances = 15000        # number of training instances from filtered-pooled MNIST

## Simulation Settings
debug_mode = 0

save_weight = 1

dump_training_stats = 1
identifier = "Fig_35"
training_stat_dump_intvl = 1000

plot_MovingAccuracy = 1
appendUntrainedStats = 1

printout_dir = "sim_printouts/MNIST/"
if supervised_hidden or supervised_output:
    dumpsim_dir = printout_dir + "Supervised/dumpsim.txt"
else:
    dumpsim_dir = printout_dir + "Inference/dumpsim.txt"

if debug_mode:
    f_handle = open(dumpsim_dir, "w+")
else:
    f_handle = None

if dump_training_stats:
    f_handle_training_stats = open(printout_dir + "Supervised/training_stats_"+ \
                                   identifier + ".txt", "w+")
    f_handle_training_stats.write("Synapse Stats during Training\n")

W_hidden = int((W_input-F_hidden) / S_hidden) + 1

num_input_neurons = W_input**2 * num_edge_maps

num_hidden_neurons = W_hidden**2 * depth_hidden_per_sublocation
num_fan_in_hidden = num_hidden_neurons * F_hidden**2 * num_edge_maps

num_output_neurons = num_categories
num_fan_in_output = num_output_neurons * num_hidden_neurons

num_neurons = num_input_neurons + num_hidden_neurons + num_output_neurons

initial_weight_input = [10] * num_input_neurons
initial_weight_hidden = [5] * num_hidden_neurons * F_hidden**2 * num_edge_maps
initial_weight_output = [5] * num_output_neurons * num_hidden_neurons
weight_vector = \
    [
        *initial_weight_input, *initial_weight_hidden, *initial_weight_output
    ]

## Initialize Connectivity
sheet_dir = "sim_printouts/MNIST/ConnectivityTable.xlsx"
input_connectivity, hidden_connectivity, output_connectivity \
, ConnectivityTable, WeightRAM, PotentialRAM, writer = \
    NetworkConnectivity.initializeNetWorkConnectivity(
        num_categories=num_categories, num_edge_maps=num_edge_maps, W_input=W_input,
        F_hidden=F_hidden, S_hidden=S_hidden,
        depth_hidden_per_sublocation=depth_hidden_per_sublocation, weight_vector=weight_vector,
        sheet_dir=sheet_dir
    )

## Initialize InhibitionScoreboard
InhibitScoreboard_lst = \
    [
        SNN.IntermapInhibitScoreboard(0, W_input, 1),
        SNN.IntermapInhibitScoreboard(1, W_hidden, 1),
        None
    ]

################################################################
# %% Loading the pooled MNIST images in shape (60000, W, W, 4)
################################################################
obj_text_pooled = codecs.open(
    "/home/usr1/cni/MNIST_filtered_pooled/pooled3x3.json", 'r', encoding='utf-8'
).read()
obj_text_train_labels = codecs.open(
    "/home/usr1/cni/MNIST_filtered_pooled/train_labels.json", 'r', encoding='utf-8'
).read()
pooled_lst = json.loads(obj_text_pooled)
train_labels = json.loads(obj_text_train_labels)

pooled = np.array(pooled_lst)
## replace negative value with 0's
pooled = pooled.clip(0, 1)

## take a few samples for testing purposes
# pooled = pooled[0:num_instances, :, :]
# train_labels = train_labels[0:num_instances]
# for j in range(num_instances):
#     imshow_pooled(pooled[j])
################################################################

#%% Map each individual pooled pixel intensity into a latency value
## and flatten to 1D array of training instances
################################################################
# LatencyArrayPoly = ItLmapping(
#     pooled_array=pooled, window=8*tau_u, kernel="polynomial",
#     alpha=0.5, poly_degree=1
# )
LatencyArray = ItLmapping(
    pooled_array=pooled, window=8*tau_u, kernel="polynomial",
    alpha=0.5, poly_degree=1
)
# hist_latency(latency_array=LatencyArrayPoly, tau_u=tau_u, num_bins=8)
# hist_latency(latency_array=LatencyArray, tau_u=tau_u, num_bins=8)
# plt.show()

## Flatten LatencyArray in edge_map-major and then row-major order into
## a 2D array of size (# of training instances, # of edge maps * W * W)
## e.g. (60000, 4*8*8)
InLatencyFlat = \
    LatencyArray.transpose(0, 3, 1, 2).reshape(LatencyArray.shape[0], -1).tolist()
################################################################

#%% Define Input & Output Patterns
################################################################
output_pattern = \
    {
        0   : num_input_neurons + num_hidden_neurons,
        1   : num_input_neurons + num_hidden_neurons + 1,
        2   : num_input_neurons + num_hidden_neurons + 2,
        3   : num_input_neurons + num_hidden_neurons + 3,
        4   : num_input_neurons + num_hidden_neurons + 4,
        5   : num_input_neurons + num_hidden_neurons + 5,
        6   : num_input_neurons + num_hidden_neurons + 6,
        7   : num_input_neurons + num_hidden_neurons + 7,
        8   : num_input_neurons + num_hidden_neurons + 8,
        9   : num_input_neurons + num_hidden_neurons + 9

    }

if plot_MovingAccuracy:
    fig_accuracy, ax_accuracy = createMovingAccuracyFigure(num_instances)

#%% Instantiate a list of SpikingNeuron objects
######################################################################################
sn_list = [None] * num_neurons
for neuron_idx in range(num_neurons):
    layer_idx = ConnectivityTable.layer_num[neuron_idx]
    if layer_idx == 0:
        inhibit_enable = 0
        sn = SNN.SpikingNeuron( layer_idx=layer_idx,
                                neuron_idx=neuron_idx,
                                location_idx=input_connectivity[neuron_idx]["pixel_idx"],
                                slice_idx=input_connectivity[neuron_idx]["edge_map_idx"],
                                inhibit_enable=inhibit_enable,
                                fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[neuron_idx],
                                fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[neuron_idx],
                                depth_causal = 1,
                                depth_anticausal = 0,
                                tau_u=tau_u,
                                tau_v=tau_v,
                                threshold=vth_input,
                                duration=duration,
                                training_on=0,
                                supervised=0
                                )
    elif layer_idx == 1:
        depth_causal = 9
        depth_anticausal = 9
        inhibit_enable = 1
        if supervised_hidden:
            training_on = 1
            supervised = 1
        sn = SNN.SpikingNeuron( layer_idx=layer_idx,
                                neuron_idx=neuron_idx,
                                location_idx=hidden_connectivity[neuron_idx - num_input_neurons]["sublocation_idx"],
                                slice_idx = hidden_connectivity[neuron_idx-num_input_neurons]["depth_idx"],
                                inhibit_enable = inhibit_enable,
                                fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[neuron_idx],
                                fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[neuron_idx],
                                depth_causal = depth_causal,
                                depth_anticausal = depth_anticausal,
                                tau_u=tau_u,
                                tau_v=tau_v,
                                threshold=vth_hidden,
                                duration=duration,
                                training_on=training_on,
                                supervised=supervised
                                )

    elif layer_idx == 2:
        depth_causal = 12
        depth_anticausal = 3
        inhibit_enable = 0
        if supervised_hidden:
            training_on = 1
            supervised = 1
        sn = SNN.SpikingNeuron( layer_idx=layer_idx,
                                neuron_idx=neuron_idx,
                                location_idx=None,
                                slice_idx = None,
                                inhibit_enable = inhibit_enable,
                                fan_in_synapse_addr=ConnectivityTable.fan_in_synapse_addr[neuron_idx],
                                fan_out_synapse_addr=ConnectivityTable.fan_out_synapse_addr[neuron_idx],
                                depth_causal = depth_causal,
                                depth_anticausal = depth_anticausal,
                                tau_u=tau_u,
                                tau_v=tau_v,
                                threshold=vth_output,
                                duration=duration,
                                training_on=training_on,
                                supervised=supervised
                                )
    sn_list[neuron_idx] = sn
######################################################################################


#%% Inter-neuron data initialization
######################################################################################
fired_neuron_list =    [
                            [
                                [] for step in range(0, sn.duration, sn.dt)
                            ]
                            for instance in range(num_instances)
                        ]   # num_instances x num_timesteps x num_fired_neurons

output_neuron_fire_info =   [
                                {
                                    "neuron_idx":   [],
                                    "time"      :   []
                                }
                                for instance in range(num_instances)
                            ]

inference_correct =     [
                            None for instance in range(num_instances)
                        ]   # num_instances

moving_window = [None] * size_moving_window
accuracy_during_training = [None] * num_instances

#%% Simulation Loop
######################################################################################
correct_cnt = 0
max_correct_cnt = 0
PreSynapticIdx_intended = \
    [
        {
            "causal"        :   [],
            "anti-causal"   :   []
        }
        for i in range(num_instances)
    ]

PreSynapticIdx_nonintended = \
    [
        {
            "causal"        :   [],
            "anti-causal"   :   []
        } for i in range(num_instances)
    ]

## Loop through instances
for instance in range(num_instances):
    ## a list of spike_info_entry that contains "fired_synapse_addr", "location_idx" and "time"
    spike_info = \
        [
            []
            for step in range(0, sn.duration, sn.dt)
        ]

    if debug_mode:
        f_handle.write("---------------Instance {} \"{}\" -----------------\n"
                       .format(instance, train_labels[instance]))

    ## Forward Pass
    for sim_point in range(0, duration, sn.dt):
        ## first check if any input synpase fires at this time step
        if sim_point in InLatencyFlat[instance]:
            synapse_indices = index_duplicate(InLatencyFlat[instance], sim_point)
            for synapse_index in synapse_indices:
                spike_info_entry = \
                    {
                        "fired_synapse_addr": synapse_index,
                        "location_idx": None,
                        "time": sim_point
                    }
                spike_info[sim_point].append(spike_info_entry)

        for i in range(num_neurons):
            sn_list[i].accumulate(sim_point=sim_point,
                                    spike_in_info=spike_info[sim_point],
                                    WeightRAM_inst=WeightRAM,
                                    instance=instance,
                                    f_handle=f_handle,
                                    InhibitScoreboard_inst=InhibitScoreboard_lst[sn_list[i].layer_idx],
                                    debug_mode=debug_mode
                                    )
            # upadate the current potential to PotentialRAM
            PotentialRAM.potential[i] = sn_list[i].v[sim_point]

            # update the list of synapses that fired at this sim_point
            if (sn_list[i].fire_cnt != -1 and sn_list[i].spike_out_info[0]["time"] == sim_point):
                spike_info[sim_point].extend(sn_list[i].spike_out_info)
                fired_neuron_record = \
                    {
                        "layer_idx"     :   sn_list[i].layer_idx,
                        "neuron_idx"    :   sn_list[i].neuron_idx
                    }
                fired_neuron_list[instance][sim_point].append(fired_neuron_record)

                ## if the fired neuron at this sim_point is in the output layer
                if (sn_list[i].layer_idx == 2):
                    output_neuron_fire_info[instance]["neuron_idx"].append(sn_list[i].neuron_idx)
                    output_neuron_fire_info[instance]["time"].append(sim_point)
    ## End of one Forward Pass

    # ## copy the sublocation_buffer to sublocation_buffer_prev
    # ## in the output layer neuron's spike_in_cache
    # recorded_sublocations = sn_list[num_input_neurons+num_hidden_neurons].spike_in_cache.sublocation_buffer
    # if debug_mode:
    #     f_handle.write("Recorded sublocation_idx: {}\n".format(recorded_sublocations))
    # for output_neuron_idx in range(num_input_neurons + num_hidden_neurons, num_neurons):
    #     sn_list[output_neuron_idx].spike_in_cache.latchSublocationBufferPrev()

    ## At the end of the Forward Pass, inspect output-layer firing info
    if len(output_neuron_fire_info[instance]["neuron_idx"]) > 0:
        # find the minimum of firing time and its corresponding list index
        min_fire_time = min(output_neuron_fire_info[instance]["time"])
        list_min_idx = index_duplicate(output_neuron_fire_info[instance]["time"], min_fire_time)
        f2f_neuron_lst =    [
                                output_neuron_fire_info[instance]["neuron_idx"][list_idx]
                                for list_idx in list_min_idx
                            ]
        if output_pattern[train_labels[instance]] in f2f_neuron_lst:
            f2f_neuron_idx = output_pattern[train_labels[instance]]
        else:
            f2f_neuron_idx = f2f_neuron_lst[0]

        ## find the non-F2F neurons that fired within separation window
        non_f2f_neuron_lst = [
            neuron_idx
            for list_idx, neuron_idx in enumerate(output_neuron_fire_info[instance]["neuron_idx"])
            if output_neuron_fire_info[instance]["time"][list_idx] - min_fire_time >= 0
               and output_neuron_fire_info[instance]["time"][list_idx] - min_fire_time <= separation_window
               and neuron_idx != f2f_neuron_idx
        ]

    else:
        min_fire_time = None
        list_min_idx = None
        f2f_neuron_lst = []
        f2f_neuron_idx = None
        non_f2f_neuron_lst = []

    if instance == 0:
        moving_accuracy = 0
    else:
        moving_accuracy = accuracy_during_training[instance-1]
    correct_cnt = SNN.combined_RSTDP_BRRC(
                    sn_list=sn_list, instance=instance, inference_correct=inference_correct,
                    num_fired_output=len(output_neuron_fire_info[instance]["neuron_idx"]),
                    supervised_hidden=supervised_hidden, supervised_output=supervised_output,
                    f_handle=f_handle,
                    PreSynapticIdx_intended=PreSynapticIdx_intended,
                    PreSynapticIdx_nonintended=PreSynapticIdx_nonintended,
                    desired_ff_idx=output_pattern[train_labels[instance]],
                    min_fire_time = min_fire_time,
                    f2f_neuron_lst=f2f_neuron_lst,
                    non_f2f_neuron_lst=non_f2f_neuron_lst,
                    f2f_neuron_idx=f2f_neuron_idx,
                    WeightRAM=WeightRAM,
                    moving_accuracy=moving_accuracy, accuracy_th=accuracy_th,
                    correct_cnt=correct_cnt,
                    num_causal_output = 12, num_anticausal_output = 3,
                    num_causal_hidden = 6, num_anticausal_hidden=6,
                    debug_mode=debug_mode
    )

    print("Instance {}: Label is {}; Inference Correct: {}"
          .format(instance, train_labels[instance], inference_correct[instance]))

    ## record training accuracy after each training instance
    if instance == 0:
        moving_window[0] = inference_correct[instance]
    elif instance > 0:
        moving_window = \
            [inference_correct[instance]] + moving_window[0:-1]

    accuracy_during_training[instance] = \
        getTrainingAccuracy(moving_window)
    if plot_MovingAccuracy:
        appendAccuracy(ax_accuracy, instance, accuracy_during_training[instance])
    if correct_cnt > max_correct_cnt:
        max_correct_cnt = correct_cnt
    if debug_mode:
        f_handle.write("Succesive correct count: {}\n".format(correct_cnt))
        f_handle.write("-------------------------------------------------\n")

    if correct_cnt == stop_num and (supervised_hidden or supervised_output):
        print("Supervised Training at step {} reached {} successive correct inferences"
              .format(instance, correct_cnt))
        break

    ## clear the state varaibles of sn_list and PotentialRAM
    InhibitScoreboard_lst[0].clearScoreboard()
    InhibitScoreboard_lst[1].clearScoreboard()
    PotentialRAM.clearPotential()
    for i in range(num_neurons):
        sn_list[i].clearStateVariables()

    ## dump training statistics
    if dump_training_stats and \
            ((instance+1) % training_stat_dump_intvl) == 0:
        getUntrainedPerc(instance=instance, WeightRAM_inst=WeightRAM, W_hidden=W_hidden,
                         num_categories=num_categories,
                         num_input_neurons=num_input_neurons, num_fan_in_hidden=num_fan_in_hidden,
                         num_fan_in_output=num_fan_in_output, appendUntrainedStats=appendUntrainedStats,
                         f_handle_training_stats=f_handle_training_stats, ax_accuracy=ax_accuracy
                         )

if debug_mode:
    f_handle.write("Maximum successive correct count:{}\n".format(max_correct_cnt))


print("moving accuracy = \n")
print(*("{0:4.3f}, ".format(k) for k in accuracy_during_training if k != None))
print("Supervised Training stops at Instance {}"
        .format(instance))

#%% Dump initial weight vector and final weightRAM
if debug_mode:
    f_handle.write("***************************Weight Change*********************\n")
    f_handle.write("Synapse\t\t InitialWeight\t\t FinalWeight\t\t\n")

    for synapse_addr in WeightRAM.synapse_addr:
        f_handle.write("{}\t\t {}\t\t\t {}\t\t\n"
                    .format(synapse_addr, weight_vector[synapse_addr], WeightRAM.weight[synapse_addr]))
    f_handle.write("************************************************************\n")
    f_handle.close()

print("Maximum successive correct count:{}\n".format(max_correct_cnt))
print("End of Program!")

if plot_MovingAccuracy:
    fig_accuracy.savefig(printout_dir+"Supervised/"+identifier,quality=100, optimize=True)

if save_weight:
    saveWeightRAM(instance, WeightRAM, identifier)
