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
import torchvision.transforms as transforms
import skimage.transform as skTrans


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

def save3Dimage_numpy(img3d, path):
        img3d = img3d.squeeze()
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
        self.dir_ct_label = os.path.join(opt.dataroot, opt.phase + 'ct_labels')  # create a path '/path/to/data/ct'
        self.dir_mr_label = os.path.join(opt.dataroot, opt.phase + 'mr_labels')  # create a path '/path/to/data/mr'

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
        
        self.margin = 10

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
        mr_path_label = self.mr_paths_label[index_mr]

        if(self.opt.four_labels):
            labels_translate = {
                0 : 0,      # Background
                205 : 1,    # MYO
                420 : 2,    # LAC
                500 : 3,    # LVC
                550 : 0,    # RAC
                600 : 0,    # RVC
                820 : 4,    # AA
                850 : 0,    # pulmonary artery
            }
        else:
            labels_translate = {
                0 : 0,      # Background
                205 : 1,    # MYO
                420 : 2,    # LAC
                500 : 3,    # LVC
                550 : 4,    # RAC
                600 : 5,    # RVC
                820 : 6,    # AA
                850 : 7,    # pulmonary artery
            }


        ct_label = np.load(ct_path_label)['arr_0']
        ct_img = np.load(ct_path)['arr_0']
        mr_label = np.load(mr_path_label)['arr_0']
        mr_img = np.load(mr_path)['arr_0']

        # Scale images to size (256,256,crop_size_z)
        # ct_scale = ct_img.shape[1]/256
        # mr_scale = mr_img.shape[1]/256
        # ct_img = skTrans.resize(ct_img, (ct_img.shape[0], int(ct_img.shape[1]/ct_scale), int(ct_img.shape[2]/ct_scale), int(ct_img.shape[3]/ct_scale)), order=1, preserve_range=True,  anti_aliasing=True)
        # mr_img = skTrans.resize(mr_img, (mr_img.shape[0], int(mr_img.shape[1]/mr_scale), int(mr_img.shape[2]/mr_scale), int(mr_img.shape[3]/mr_scale)), order=1, preserve_range=True,  anti_aliasing=True)
        # mr_label = skTrans.resize(mr_label, (mr_label.shape[0], int(mr_label.shape[1]/mr_scale), int(mr_label.shape[2]/mr_scale), int(mr_label.shape[3]/mr_scale)), order=0, preserve_range=True, anti_aliasing=False)
        # ct_label = skTrans.resize(ct_label, (ct_label.shape[0], int(ct_label.shape[1]/ct_scale), int(ct_label.shape[2]/ct_scale), int(ct_label.shape[3]/ct_scale)), order=0, preserve_range=True, anti_aliasing=False)

        if self.opt.preprocess != "none":

            # ct_box = [slice(np.min(indexes), np.max(indexes) + 1) for indexes in np.where(ct_label>0)]
            # for i in range(1, 3):
            #     if (ct_box[i].stop - ct_box[i].start) < self.opt.crop_size:
            #         ct_box[i] = slice(ct_box[i].start, ct_box[i].start + self.opt.crop_size)
            # if (ct_box[3].stop - ct_box[3].start) < self.opt.crop_size_z:
            #     ct_box[3] = slice(ct_box[3].start, ct_box[i].start + self.opt.crop_size_z)

            # ct_box = tuple(ct_box)
            # ct_img = ct_img[ct_box]
            # ct_label = ct_label[ct_box]


            x = random.randint(0, np.maximum(0, ct_img.shape[1] - self.opt.crop_size - self.margin))
            y = random.randint(0, np.maximum(0, ct_img.shape[2] - self.opt.crop_size - self.margin))
            z = random.randint(0, np.maximum(0, ct_img.shape[3] - self.opt.crop_size_z - self.margin))

            if not self.opt.no_crop:
                ct_img = torch.from_numpy(ct_img[:,x:x+self.opt.crop_size, y:y+self.opt.crop_size, z:z+self.opt.crop_size_z])
                ct_label = torch.from_numpy(ct_label[:,x:x+self.opt.crop_size, y:y+self.opt.crop_size, z:z+self.opt.crop_size_z])

            x = random.randint(0, np.maximum(0, mr_img.shape[1] - self.opt.crop_size - self.margin))
            y = random.randint(0, np.maximum(0, mr_img.shape[2] - self.opt.crop_size - self.margin))
            z = random.randint(0, np.maximum(0, mr_img.shape[3] - self.opt.crop_size_z - self.margin))

            if not self.opt.no_crop:
                mr_img = torch.from_numpy(mr_img[:,x:x+self.opt.crop_size, y:y+self.opt.crop_size, z:z+self.opt.crop_size_z])
                mr_label = torch.from_numpy(mr_label[:,x:x+self.opt.crop_size, y:y+self.opt.crop_size, z:z+self.opt.crop_size_z])

        else:
            ct_img = self.transform_ct(torch.from_numpy(ct_img))
            ct_label = self.transform_ct(torch.from_numpy(ct_label))
            mr_img = self.transform_mr(torch.from_numpy(mr_img))
            mr_label = self.transform_mr(torch.from_numpy(mr_label))

        # FIX 421 to 420 - dataset bug
        mr_label[mr_label == 421] = 420
        # End of FIX

        #change labels to 0-7 (or 0-4 in case of four labels)
        for key in labels_translate:
            mr_label[mr_label == key] = labels_translate[key]
            ct_label[ct_label == key] = labels_translate[key]

        return {'ct': ct_img, 'mr': mr_img, 'ct_paths': ct_path, 'mr_paths': mr_path, 'ct_label': ct_label, 'mr_label': mr_label}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.ct_size, self.mr_size)