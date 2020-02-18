
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import codecs, json 
#######################################################################################

def genGabor(size, gamma, sigma, lambd, theta):
    # size = (5,5), gamma=1, sigma=3, lambd=5
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    g = 1.2 * np.exp(-(x1**2 + gamma**2 * y1**2) / (2*sigma**2)) * np.cos(2*np.pi*x1/lambd)
    return g

def genGaborBank(size, gamma, sigma, lambd, theta_lst):
    g_bank =\
        [
            {
                "filter"    :   None,
                "theta"     :   None
            } for i in range(len(theta_lst))
        ]
    for i in range(len(theta_lst)):
        g = genGabor(size=size, gamma=gamma, sigma=sigma, lambd=lambd, theta=theta_lst[i])
        g_bank[i]["filter"] = g
        g_bank[i]["theta"] = theta_lst[i]
    
    return g_bank

def imshowGaborBank(g_bank):
    num_thetas = len(g_bank)
    fig, ax = plt.subplots(1, num_thetas, figsize=(15, 5))
    fig.subplots_adjust(left=0.05, right=0.95)
    for i in range(num_thetas):
        ax[i].imshow(g_bank[i]["filter"], cmap='gray')
        ax[i].set_title(r"$\theta = $" + "{0:3.2f}".format(g_bank[i]["theta"] / np.pi) + r"$\pi$", fontsize=10)


def convolve2D(X, kernel, stride=1, array_type=np.float32):
    # X is a  2D numpy array
    # kenrel is a 2D numpy array 
    X_filtered = np.zeros(
        ((X.shape[0] - kernel.shape[0]) // stride + 1,
        (X.shape[0] - kernel.shape[0]) // stride + 1)
    )

    for x in range(X_filtered.shape[0]):
        for y in range(X_filtered.shape[1]):
            X_top_left_idx = x*X.shape[1]*stride + y*stride
            X_row_idx_start = X_top_left_idx // X.shape[1]
            X_col_idx_start = X_top_left_idx % X.shape[1]
            X_filtered[x][y] =\
                np.sum(
                    np.multiply(
                    X[X_row_idx_start:X_row_idx_start+kernel.shape[0], X_col_idx_start:X_col_idx_start+kernel.shape[1]],
                    kernel
                    )
                )

    return X_filtered.astype(array_type)


def convolveWithGabor(X, g_bank, stride=1, array_type=np.float32):
    # X_training is a list of (60000, 28, 28) numpy arrays
    # g_bank is a list of dictionary with fields "filter" and "theta" of length num_edge_maps
    # where "filter" field holds a numpy array
    if (X.shape[1] - g_bank[0]["filter"].shape[0]) % stride != 0:
        print("Error when calling convolveWithGabor(): dimension mismatch between X[0], g_bank[0] and stride")
        exit(1)
    if (X.shape[2] - g_bank[0]["filter"].shape[1]) % stride != 0:
        print("Error when calling convolveWithGabor(): dimension mismatch between X[1], g_bank[1] and stride")
        exit(1)
    
    num_edge_maps = len(g_bank)
    
    X_filtered =\
        np.zeros(
            (
                X.shape[0], 
                (X.shape[1] - g_bank[0]["filter"].shape[0])//stride + 1, 
                (X.shape[2] - g_bank[0]["filter"].shape[0])//stride + 1,
                num_edge_maps
            ),
            dtype=array_type
        )

    for i in range(X.shape[0]):
       for edge_idx in range(num_edge_maps):
           X_filtered[i, :, :, edge_idx] = convolve2D(X[i, :, :], g_bank[edge_idx]["filter"], stride)
    
    return X_filtered


def imshowEdgeMaps(X, X_filtered, num_edge_maps=4):
    # X is a 2D numpy array
    # X_filtered is a 3D numpy array with the 3rd dimension being edge_map_idx
    fig, ax = plt.subplots(nrows=num_edge_maps, ncols=2, figsize=(15,8))

    for axis in ax.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
        
    gs = ax[0,0].get_gridspec()
    for axis in ax[0:, 0]:
        axis.remove()
    axbig = fig.add_subplot(gs[0:, 0])
    axbig.set_xticks([])
    axbig.set_yticks([])
    fig.tight_layout()

    axbig.imshow(X, cmap='gray')
    for edge_idx in range(num_edge_maps):
        ax[edge_idx, 1].imshow(X_filtered[:, :, edge_idx], cmap='gray')

def normalizeGaborFiltered(X_filtered):
    # X_filtered is a list of dictionaries with field "edge_maps" and "label"
    # where "edge_maps" holds a 3D array with the 3rd dimension being edge_map
    # and   "label" holds an integer
    # for i in range(len(X_filtered)):
    #     X_filtered[i]["edge_maps"] = X_filtered[i]["edge_maps"] / np.amax(X_filtered[i]["edge_maps"])
    for i in range(X_filtered.shape[0]):
        X_filtered[i, :, :, :] = X_filtered[i, :, :, :] / np.amax(X_filtered[i, :, :, :])

def pooling(X_filtered, W, stride=None, array_type=np.float32):
    # X_filtered is a list of dictionaries with field "edge_maps" and "label"
    # where "edge_maps" holds a 3D array with the 3rd dimension being edge_map
    # and   "label" holds an integer
    if stride == None:  stride = W
    if (X_filtered.shape[1] - W) % stride != 0:
        print("Error when calling pooling(): dimension 0 {} of edge maps in X_filtered does not match specified W {} and stride {}"
            .format(X_filtered.shape[1], W, stride))
        exit(1)
    if (X_filtered.shape[2] - W) % stride != 0:
        print("Error when calling pooling(): dimension 1 {} of edge maps in X_filtered does not match specified W {} and stride {}"
            .format(X_filtered.shape[2], W, stride))
        exit(1)
    num_edge_maps = X_filtered.shape[3]

    X_pooled =\
        np.zeros(
            (
                X_filtered.shape[0], 
                (X_filtered.shape[1] - W)//stride + 1, 
                (X_filtered.shape[2] - W)//stride + 1, 
                num_edge_maps
            ),
            dtype=array_type
        )

    for i in range(X_filtered.shape[0]):
        for edge_map_idx in range(num_edge_maps):        
            for x in range((X_filtered.shape[1] - W)//stride + 1):
                for y in range((X_filtered.shape[2] - W)//stride + 1):
                    X_top_left_idx = x*X_filtered.shape[2]*stride + y*stride
                    X_row_idx_start = X_top_left_idx // X_filtered.shape[2]
                    X_col_idx_start = X_top_left_idx % X_filtered.shape[2]
                    X_pooled[i, x, y, edge_map_idx] = \
                        np.amax(
                            X_filtered[i, X_row_idx_start:X_row_idx_start+W, X_col_idx_start:X_col_idx_start+W, edge_map_idx]
                        )
    return(X_pooled)


def imshowEdgeMapAndPooled(X, X_filtered, X_pooled, num_edge_maps=4):
    # X is a 2D numpy array
    # X_filtered is a 3D numpy array with the 3rd dimension being edge_map_idx
    # X_pooled is a 3D numpy array with the 3rd dimension being edge_map_idx
    fig, ax = plt.subplots(nrows=num_edge_maps, ncols=3, figsize=(15,8))
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.2, hspace=0.2)
    for axis in ax.flatten():
        axis.set_xticks([])
        axis.set_yticks([])

    gs = ax[0,0].get_gridspec()
    for axis in ax[0:, 0]:
        axis.remove()
    axbig = fig.add_subplot(gs[0:, 0])
    axbig.set_xticks([])
    axbig.set_yticks([])
    fig.tight_layout()

    axbig.imshow(X, cmap='gray')
    
    for edge_idx in range(num_edge_maps):
        ax[edge_idx, 1].imshow(X_filtered[:, :, edge_idx], cmap='gray')
        ax[edge_idx, 2].imshow(X_pooled[:, :, edge_idx], cmap='gray')

#%% generate Gabor filter bank
#################################################################################################
theta_lst = [np.pi/2, np.pi/4, 0, 3*np.pi/4]
g_bank = genGaborBank(size=(5,5), gamma=1, sigma=3, lambd=5, theta_lst=theta_lst)

#################################################################################################

#%% read in MNIST training images and reshape
#################################################################################################
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
train_images = train_images[0:3, :, :]
train_labels = train_labels[0:3]

X_training = train_images

#################################################################################################

#%%  get edge maps of all the training images
#################################################################################################
X_filtered = convolveWithGabor(X_training, g_bank, stride=1)
normalizeGaborFiltered(X_filtered)
X_pooled = pooling(X_filtered, W=3)

# imshowEdgeMaps(X_training[12]["image"], X_filtered[0]["edge_maps"])
# imshowEdgeMapAndPooled(X_training[1,:,:], X_filtered[1,:,:,:], X_pooled[1,:,:,:])
# plt.show()

#%% save filtered_normalized edge maps (X_filtered) and pooled edge maps (X_pooled)
#################################################################################################
# with open("./MNIST_filtered_pooled/filtered_normalized.json", "w") as fout0:
#     json.dump(X_filtered, fout0)
# with open("./MNIST_filtered_pooled/pooled.json", "w") as fout1:
#     json.dump(X_filtered, fout1)

X_filtered_lst = X_filtered.tolist()
X_pooled_lst = X_pooled.tolist()
train_labels_lst = train_labels.tolist()
json.dump(
    X_filtered_lst, 
    codecs.open("./MNIST_filtered_pooled/filtered_normalized.json", 'w', encoding='utf-8'),
    separators=(',',':'), sort_keys=True, indent=4
    )
json.dump(
    X_pooled_lst, 
    codecs.open("./MNIST_filtered_pooled/pooled3x3.json", 'w', encoding='utf-8'),
    separators=(',',':'), sort_keys=True, indent=4
    )
json.dump(
    train_labels_lst, 
    codecs.open("./MNIST_filtered_pooled/train_labels.json", 'w', encoding='utf-8'),
    separators=(',',':'), sort_keys=True, indent=4
    )

#################################################################################################

print("End of Program")