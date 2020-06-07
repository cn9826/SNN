from operator import itemgetter
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import codecs
import json


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

#%%
# A_pos = 1 
# tau_pos = 10
# for s in range(-30, 1, 1):
#     deltaWeight = round(A_pos * math.exp(s/tau_pos))
#     print("s = {}\t\t deltaWeight = {}".format(s, deltaWeight))

#%%
# list_2d = \
#     [
#         [1, 2, 3], [4, 5, 6], [7, 8, 9]
#     ]
# list_1d = [ x for sub_list in list_2d for x in sub_list]
# print(list_1d[0:-1])

# lst1 = [2, None, 3, None]

lst1 = [0, 1, 2]
hidden_or_output_str = "Hidden"
print("{0:7s} Neuron could not find an anti-causal in-spike entry!"
                .format(hidden_or_output_str))
#%%
# def createMovingAccuracyFigure(num_instances):
#     fig, ax = plt.subplots(figsize=(14, 7))
#     xticklabel_list = ['{}'.format(i) for i in range(0, num_instances, num_instances//10)]
    
#     ax.set_xlim(0, num_instances+1)
#     ax.set_xticks(range(0, num_instances, num_instances//10))
#     ax.set_xticklabels(xticklabel_list)
#     ax.set_xlabel('Instance Index', fontsize=12)

#     ax.set_ylabel('Moving Accuracy During Training')
#     ax.set_ylim(0, num_instances)

#     ax.grid(which='major', axis='y')
#     ax.hlines(y=50, xmin=0, xmax=num_instances-1, lw=2, color='0.5', linestyle='dashed')
#     return (fig, ax)

# def animate(ax, instance, y):
#     ax.scatter(instance, y, marker='o', color='r', s=5)
#     plt.pause(0.001)

# fig, ax = createMovingAccuracyFigure(1000)

# instance_lst = [x for x in range(0,1000)]
# y_lst = [x for x in instance_lst]

#%%
# arr = np.array(
#     [[
#         [0, 1, 2],
#         [3, 4, 5],
#     ],
#     [
#         [6, 7, 8],
#         [9, 10, 11]
#     ]]
# )
#
# print(arr.shape)
#
# arr_reshaped = arr.reshape(2, 6)


