import time
import os
import random
import math
import torch
import numpy as np
#from tqdm import tqdm
from skimage import io, color
from skimage.transform import rescale


def distance_batch(x, X):
    dist = torch.cdist(x,X,p=2)
    return dist

def gaussian(dist, bandwidth):
    """ compute weights of each point according to distance 
    :param: dist: distance btw x and X
    :param: bandwidth: std deviation of the gaussian
    :returns: 1-d np-array of weights of size len(X) """
    # weight decreases when point is far away from mean
    weight = np.exp(-(dist*dist)/(2* bandwidth**2))
    return weight

def update_point_batch(weight, X):
    denom = torch.sum(weight,dim=0)
    mean = torch.matmul(weight,X)/denom.unsqueeze(1)
    return mean

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    dist = distance_batch(X, X)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        print(_)
        X = meanshift_step_batch(X)   # fast implementation
    return X

def create_colors():
    np.random.seed(42)
    colors = np.random.rand(200, 3) * [100, 128, 128]
    colors_lab = color.rgb2lab(colors)
    return colors_lab

scale = 0.2    # downscale the image to run faster 0.25

# Load image and convert it to CIELAB space
image = rescale(io.imread('./input/img11_inpainting_output.png'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors_ = create_colors()
centroids, labels = np.unique((X / 50).round(), return_inverse=True, axis=0)
result_image = colors_[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
image_name = 'img11_inpainting_output'
io.imsave(f'./results/resultMeanshift_{image_name}.png', result_image)