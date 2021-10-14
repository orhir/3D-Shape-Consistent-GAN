"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.colors as clr


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:http://localhost:8097/
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save3Dimage_numpy(img3d, path, label=None):

        img_shape = img3d.shape
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig, ax3 = plt.subplots(1)
        # ax1.imshow(img3d[img_shape[0]//2, :, :], cmap="gray")
        # ax1.title.set_text("Middle X")
        # ax1.axis('off') 
        # ax2.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # ax2.title.set_text("Middle Y")
        # ax2.axis('off') 
        cmap = clr.ListedColormap(["black", "black", "black", "black", "lightskyblue", "black", "black", "black", "black", "lightsalmon", "papayawhip", "lightpink"])
        print("labels", np.unique(img3d))
        if "_B" in label:
            img3d = img3d[:, 50:230, :]
        ax3.imshow(img3d[:, :, img_shape[2]//2-5], cmap='gray')

        ax3.axis('off') 

        plt.savefig(path)
        plt.close('all')

def save3D_3slices_numpy(img3d, path):

        img_shape = img3d.shape
        fig, ((ax1_a, ax1_b, ax1_c), (ax2_a, ax2_b, ax2_c), (ax3_a, ax3_b, ax3_c)) = plt.subplots(3, 3)
        ax1_a.imshow(img3d[img_shape[0]//4, :, :], cmap="gray")
        ax1_a.title.set_text("1/4")
        ax1_a.axis('off') 
        ax1_b.imshow(img3d[2*(img_shape[0]//4), :, :], cmap="gray")
        ax1_b.title.set_text("1/2")
        ax1_b.axis('off') 
        ax1_c.imshow(img3d[3*(img_shape[0]//4), :, :], cmap="gray")
        ax1_c.title.set_text("3/4")
        ax1_c.axis('off') 
        ax2_a.imshow(img3d[:, img_shape[1]//4, :], cmap="gray")
        ax2_a.title.set_text("1/4")
        ax2_a.axis('off') 
        ax2_b.imshow(img3d[:, 2*(img_shape[1]//4), :], cmap="gray")
        ax2_b.title.set_text("1/2")
        ax2_b.axis('off') 
        ax2_c.imshow(img3d[:, 3*(img_shape[1]//4), :], cmap="gray")
        ax2_c.title.set_text("3/4")
        ax2_c.axis('off') 
        ax3_a.imshow(img3d[:, :, img_shape[2]//4], cmap="gray")
        ax3_a.title.set_text("1/4")
        ax3_a.axis('off') 
        ax3_b.imshow(img3d[:, :, 2*(img_shape[2]//4)], cmap="gray")
        ax3_b.title.set_text("1/2")
        ax3_b.axis('off') 
        ax3_c.imshow(img3d[:, :, 3*(img_shape[2]//4)], cmap="gray")
        ax3_c.title.set_text("3/4")
        ax3_c.axis('off') 

        plt.savefig(path)
        plt.close('all')

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, d = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
