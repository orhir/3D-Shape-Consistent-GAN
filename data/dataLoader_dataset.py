import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchio as tio
import numpy as np
import nibabel as nib
import torch.nn.functional as nnf
import torch
import matplotlib.pyplot as plt


def save3Dimage(img3d, img_shape, path):
        img3d = torch.squeeze(img3d, 0)
        img_shape = img3d.shape

        plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2]//2], cmap="gray")
        # a1.set_aspect(ax_aspect)

        plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # a2.set_aspect(sag_aspect)

        plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap="gray")
        # a3.set_aspect(cor_aspect)

        plt.savefig(path)

def save3Dimage_numpy(img3d, img_shape, path):

        plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2]//2], cmap="gray")
        # a1.set_aspect(ax_aspect)

        plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # a2.set_aspect(sag_aspect)

        plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap="gray")
        # a3.set_aspect(cor_aspect)

        plt.savefig(path)


class DataLoaderDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ct = os.path.join(opt.dataroot, opt.phase + 'ct')  # create a path '/path/to/data/ct'
        self.dir_mr = os.path.join(opt.dataroot, opt.phase + 'mr')  # create a path '/path/to/data/mr'
        self.dir_ct_label = os.path.join(opt.dataroot, 'trainct_lables')  # create a path '/path/to/data/ct'
        self.dir_mr_label = os.path.join(opt.dataroot, 'trainmr_lables')  # create a path '/path/to/data/mr'

        self.ct_paths = sorted(make_dataset(self.dir_ct, opt.max_dataset_size))   # load images from '/path/to/data/ct'
        self.mr_paths = sorted(make_dataset(self.dir_mr, opt.max_dataset_size))    # load images from '/path/to/data/mr'
        self.ct_paths_label = sorted(make_dataset(self.dir_ct_label, opt.max_dataset_size))   # load images from '/path/to/data/ct'
        self.mr_paths_label = sorted(make_dataset(self.dir_mr_label, opt.max_dataset_size))    # load images from '/path/to/data/mr'

        self.ct_size = len(self.ct_paths)  # get the size of dataset ct
        self.mr_size = len(self.mr_paths)  # get the size of dataset mr
        self.ct_size_label = len(self.ct_paths_label)  # get the size of dataset ct
        self.mr_size_label = len(self.mr_paths_label)  # get the size of dataset mr
        mrtoct = self.opt.direction == 'mrtoct'
        input_nc = self.opt.output_nc if mrtoct else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if mrtoct else self.opt.output_nc      # get the number of channels of output image
        self.transform_ct = get_transform(self.opt)
        self.transform_mr = get_transform(self.opt)


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains ct, mr, ct_paths and mr_paths
            ct (tensor)       -- an image in the input domain
            mr (tensor)       -- its corresponding image in the target domain
            ct_paths (str)    -- image paths
            mr_paths (str)    -- image paths
        """
        ct_path = self.ct_paths[index % self.ct_size]  # make sure index is within then range
        ct_path_label = self.ct_paths_label[index % self.ct_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_mr = index % self.mr_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_mr = random.randint(0, self.mr_size - 1)
        mr_path = self.mr_paths[index_mr]
        mr_path_label = self.mr_paths[index_mr]

        ct_subject = tio.Subject(
            t1 = tio.ScalarImage(ct_path),
            label = tio.LabelMap(ct_path_label)
        )         
        
        mr_subject = tio.Subject(
            t1 = tio.ScalarImage(mr_path),
            label = tio.LabelMap(mr_path_label)
        ) 


        # apply image transformation
        # ------------------------------------------------
        ct_subject_transformed = self.transform_ct(ct_subject)
        mr_subject_transformed = self.transform_mr(mr_subject)

        ct = ct_subject_transformed.t1.data
        mr = mr_subject_transformed.t1.data
        ct_label = ct_subject_transformed.label.data
        mr_label = mr_subject_transformed.label.data

        # print(ct.shape)
        # print(mr.shape)
        # print(ct_label.shape)
        # print(mr_label.shape)

        # ------------------------------------------------
        return {'ct': ct, 'mr': mr, 'ct_paths': ct_path, 'mr_paths': mr_path, 'ct_label': ct_label, 'mr_label': mr_label}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.ct_size, self.mr_size)