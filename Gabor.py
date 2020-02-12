
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
#######################################################################################
## cv2. getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)

# sigma is the standard deviation of the Gaussian function, also known as the bandwidth (measured in pi*pixel?)  
#   large bandwidth the envelope increases, allowing more strips (number of strips increases)

# theta controls the vertical orientation of the Gabor function, clockwise offset

# lambd is the wavelength, controls the width of the strips of Gabor function (measured in pi*pixel?)  
#   larger lambda produces thicker strip

# gamma controls the aspect ratio 
#   larger gamma produces shorter height

# psi is the phase offset of teh sinusoidal function --> set to 0 
#######################################################################################
# def build_filters(ksize, num_thetas, sigma=1, lambd=2*np.pi, gamma=0.8):
#     filters = []
#     for theta in np.arange(0, np.pi, np.pi / num_thetas):
#         kern = \
#             cv2.getGaborKernel(ksize=(ksize, ksize), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma)
#         # kern /= 1 * kern.sum()
#         filters.append(kern)
#     return filters

# filter_banks = build_filters(ksize=7, num_thetas=1)


# def single_filter(ksize, theta=0, sigma=1, lambd=2*np.pi, gamma=0.8):
#     kern = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma)
#     return kern


# gamma = 0.4
# ksize = 5

# sigma_lst = [x for x in np.arange(1,2,0.1)]
# lambd_lst = [x for x in np.arange(1, 5, 1)]

# num_rows = len(lambd_lst)
# num_cols = len(sigma_lst)
# fig, ax = plt.subplots(num_rows, num_cols, figsize=(20,20))
# fig.subplots_adjust(left=0.125, right=0.2, wspace=0.5, hspace=0.3)
# fig.tight_layout()

# for row in range(num_rows):
#     for col in range(num_cols):
#         kern = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=sigma_lst[col], theta=0, lambd=lambd_lst[row], gamma=gamma)
#         ax[row][col].axes.get_xaxis().set_visible(False)
#         ax[row][col].axes.get_yaxis().set_visible(False)

#         ax[row][col].imshow(kern, cmap='gray')
#         ax[row][col].set_title(r"$\lambda=$" + "{0:2.1f} ".format(lambd_lst[row]) + r"$\sigma=$" + "{0:2.1f}".format(sigma_lst[col]), fontsize=8)


# def genGabor(size, omega, theta, func=np.sin, K=np.pi):
#     radius = (int(size[0]/2.0), int(size[1]/2.0))
#     [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

#     x1 = x * np.cos(theta) + y * np.sin(theta)
#     y1 = -x * np.sin(theta) + y * np.cos(theta)

#     gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2/(8*K**2)) * (4*x1**2 + y1**2)
#     sinusoid = func(omega * x1) * np.exp(K**2 / 2)
#     gabor = gauss * sinusoid
#     return gabor

def genGabor(size, gamma, sigma, lambd, theta):
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    g = np.exp(-(x1**2 + gamma**2 * y1**2) / (2*sigma**2)) * np.cos(2*np.pi*x1/lambd)
    return g


plt.imshow(genGabor(size=(5, 5), gamma=1, sigma=3, lambd=5, theta=3*np.pi/4), cmap='gray')
plt.show()