import SNN
import numpy as np
import math
import matplotlib.pyplot as plt
# import NetworkConnectivity
import codecs
import json


def ItLmapping(pooled_array, duration, kernel, alpha=0.5, poly_degree=2):
    # the intensity-to-latency mapping function
    # mapping kernels are "exponential" and "polynomial"

    # pooled_array is a ndnumpy array with shape(# of instances, W, W, 4) whose values
    # have been normalized to 1
    # kernel is one of the: "exponential" or "polynomial"
    # duration is an int indicating the duration of one forward pass in SNN
    # alpha is the scaling factor in front of the exponential term
    # poly_degree is the degree of polynomial should "polynomial" be chosen as the mapping kernel

    kernel_lst = ["exponential", "polynomial"]

    if kernel not in kernel_lst:
        print("Error when calling ItLmapping: specified kernel {} is not supported!"
              .format(kernel))
        print("Defaulting kernel to be \"exponential\"!")
        kernel = "exponential"

    if kernel == "exponential":
        beta = math.log((duration + 1) / alpha)
        LatencyArray = \
            np.rint(-1 * alpha * np.exp(beta*pooled_array) + duration + 1)
    elif kernel == "polynomial":
        LatencyArray = \
            np.rint(duration * (-1 * np.power(pooled_array, poly_degree) + 1))

    return LatencyArray


def hist_latency(latency_array, tau_u, num_bins=8):
    latency_array_tau = latency_array / tau_u
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xticks(range(0, num_bins + 1))
    xticklabel_list = [r'{}$\tau_u$'.format(i) for i in range(0, num_bins + 1)]
    ax.set_xticklabels(xticklabel_list)
    ax.hist(latency_array_tau.flatten(), bins=num_bins, edgecolor='k')
    ax.set_ylim(0,5e5)
    ax.set_title('Stimulus Spike Latency Distribution -- filtered and pooled MNIST',
                 fontsize=18, fontweight='bold')
# %% define SNN parameters
################################################################
duration = 80
tau_u = 8
################################################################
# %% Loading the pooled MNIST images in shape (60000, W, W, 4)
################################################################
obj_text_pooled = codecs.open(
    "./MNIST_filtered_pooled/pooled3x3.json", 'r', encoding='utf-8'
).read()
pooled_lst = json.loads(obj_text_pooled)
pooled = np.array(pooled_lst)
# replace negative value with 0's
pooled = pooled.clip(0, 1)

################################################################

# %% Map each individual pooled pixel intensity into a latency value
################################################################
LatencyArray = ItLmapping(
    pooled_array=pooled, duration=duration, kernel="exponential",
    alpha=0.5
)
# hist_latency(latency_array=LatencyArray, tau_u=tau_u, num_bins=10)
# plt.show()

################################################################
