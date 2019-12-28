import SNN
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from operator import itemgetter

#%%
## Define the categories of stimulus
def shapeMatrix(category, dimension=4, high_pix=1, low_pix=0, num_corrode=0):
    category_list = ["X", "O", "<<"]
    if not category in category_list:
        print("Error when calling shapeMatrix: category {} is not defined!".format(category))
        exit(1)
    matrix = np.full((dimension, dimension), fill_value=low_pix, dtype=int, order='C')
    if category == "O":
        matrix[0, np.array([1,2])] = high_pix
        matrix[1, np.array([0,3])] = high_pix
        matrix[2, np.array([0,3])] = high_pix
        matrix[3, np.array([1,2])] = high_pix
    elif category == "X":
        matrix[0, np.array([0,3])] = high_pix
        matrix[1, np.array([1,2])] = high_pix
        matrix[2, np.array([1,2])] = high_pix
        matrix[3, np.array([0,3])] = high_pix
    elif category == "<<":
        matrix[0, np.array([1,3])] = high_pix
        matrix[1, np.array([0,2])] = high_pix
        matrix[2, np.array([0,2])] = high_pix
        matrix[3, np.array([1,3])] = high_pix
    if num_corrode != 0:
        random_indices =  [
                            (random.randint(0,dimension-1), random.randint(0,dimension-1))
                            for i in range(num_corrode)
                        ]
        for index in random_indices:
            matrix[index] = low_pix
    
    return matrix
def imshowMatrix (fig_idx, matrix, cmap='gray'):
    extent=(0, matrix.shape[1], matrix.shape[0], 0)
    fig = plt.figure(fig_idx)
    ax1=plt.subplot(1,1,1)
    ax1.imshow(matrix, extent=extent, cmap=cmap)
    ax1.set_xticks(np.arange(0,4,1))
    ax1.set_yticks(np.arange(3,-1,-1))
    ax1.hlines(y=range(1,matrix.shape[1]), xmin=0, xmax=4, lw=2, color='0.5', linestyles='dashed')
    ax1.hlines(y=matrix.shape[1]/2, xmin=0, xmax=4, lw=4, color='0.1', linestyles='solid')

    ax1.vlines(x=range(1,matrix.shape[0]), ymin=0, ymax=4, lw=2, color='0.5', linestyles='dashed')
    ax1.vlines(x=matrix.shape[0]/2, ymin=0, ymax=4, lw=4, color='0.1', linestyles='solid')

#%%
######################################################################################
matrix = shapeMatrix(category="<<", dimension=4, high_pix=255, low_pix=200, num_corrode=0)

imshowMatrix(0, matrix)
plt.show()