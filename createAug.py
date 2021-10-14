import sys
import time
import numpy as np
import os
from PIL import Image
import random
import torchio as tio
import nibabel as nib
import torch.nn.functional as nnf
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage.transform as skTrans
from data.dataLoader_dataset import save3Dimage_numpy
from util.util import save3Dimage_numpy
import scipy
import nibabel as nib

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # print(fname, fname.endswith("nii.gz"))
            if fname.endswith(".nii.gz") or fname.endswith(".npz") or fname.endswith(".nii"):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def get_transform(params=None, convert=False, no_aug=True):
    transform_list = []

    # Augmentations:
    if not no_aug:
        print("Using Spatial Aug")
        transform_list += [tio.RandomAnisotropy()]
        max_displacement = 15, 15, 0
        spatial_transforms = {
            tio.RandomElasticDeformation(max_displacement=5, num_control_points=7, locked_borders=2) : 0.75
            ,tio.RandomAffine(scales=(0.9, 1.2), degrees=10, isotropic=True, image_interpolation='nearest'): 0.25
        }
        transform_list += [tio.OneOf(spatial_transforms)]
        transform_list += [tio.RandomElasticDeformation(max_displacement=max_displacement, locked_borders=2)]
    if convert:
        print("Rescale Intensity")
        transform_list += [tio.RescaleIntensity((-1, 1))]


    return tio.Compose(transform_list)

class DataLoaderDataset():

    def __init__(self, dataroot, no_aug=False, test=False, only_resize=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.no_aug = no_aug
        if test:
            print("Test data")
            self.dir_ct = os.path.join(dataroot, 'testct')  # create a path '/path/to/data/ct'
            self.dir_mr = os.path.join(dataroot, 'testmr')  # create a path '/path/to/data/mr'
            self.dir_ct_label = os.path.join(dataroot, 'testct_labels')  # create a path '/path/to/data/ct'
            self.dir_mr_label = os.path.join(dataroot, 'testmr_labels')  # create a path '/path/to/data/mr'
        else:
            self.dir_ct = os.path.join(dataroot, 'trainct')  # create a path '/path/to/data/ct'
            self.dir_mr = os.path.join(dataroot, 'trainmr')  # create a path '/path/to/data/mr'
            self.dir_ct_label = os.path.join(dataroot, 'trainct_labels')  # create a path '/path/to/data/ct'
            self.dir_mr_label = os.path.join(dataroot, 'trainmr_labels')  # create a path '/path/to/data/mr'

        self.ct_paths = sorted(make_dataset(self.dir_ct, float("inf")))   # load images from '/path/to/data/ct'
        self.mr_paths = sorted(make_dataset(self.dir_mr, float("inf")))    # load images from '/path/to/data/mr'
        self.ct_paths_label = sorted(make_dataset(self.dir_ct_label, float("inf")))   # load images from '/path/to/data/ct'
        self.mr_paths_label = sorted(make_dataset(self.dir_mr_label, float("inf")))   # load images from '/path/to/data/mr'

        self.ct_size = len(self.ct_paths)  # get the size of dataset ct
        self.mr_size = len(self.mr_paths)  # get the size of dataset mr
        self.ct_size_label = len(self.ct_paths_label)  # get the size of dataset ct
        self.mr_size_label = len(self.mr_paths_label)  # get the size of dataset mr
        mrtoct = 'mrtoct'
        if test or only_resize:
            self.transform_ct = get_transform(no_aug=True, convert=True)
            self.transform_mr = get_transform(no_aug=True, convert=True)
        else:
            self.transform_ct = get_transform(no_aug=False, convert=True)
            self.transform_mr = get_transform(no_aug=False, convert=True)
        self.gpu_ids = ["0", "1"]
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.only_resize = only_resize
        self.test = test
        self.aug = True

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
        # index_mr = random.randint(0, self.mr_size - 1)
        index_mr = index

        mr_path = self.mr_paths[index_mr]
        mr_path_label = self.mr_paths_label[index_mr]

        if ".npz" in ct_path:
            ct_label = np.load(ct_path_label)['arr_0']
            ct_img = np.load(ct_path)['arr_0']
            mr_label = np.load(mr_path_label)['arr_0']
            mr_img = np.load(mr_path)['arr_0']
        else:
            mr_img = nib.load(mr_path)
            mr_img = nib.as_closest_canonical(mr_img)
            mr_img = np.expand_dims(np.array(mr_img.dataobj), axis=0)
            mr_label = nib.load(mr_path_label)
            mr_label = nib.as_closest_canonical(mr_label)
            mr_label = np.expand_dims(np.array(mr_label.dataobj), axis=0)

            ct_img = nib.load(ct_path)
            ct_img = nib.as_closest_canonical(ct_img)
            ct_img = np.expand_dims(np.array(ct_img.dataobj), axis=0)
            ct_label = nib.load(ct_path_label)
            ct_label = nib.as_closest_canonical(ct_label)
            ct_label = np.expand_dims(np.array(ct_label.dataobj), axis=0)
        
        numpy_reader = lambda path: np.load(path)['arr_0'], np.eye(4)
        ct_subject = tio.Subject(
            t1 = tio.ScalarImage(tensor=ct_img),
            label = tio.LabelMap(tensor=ct_label)
        )             
        mr_subject = tio.Subject(
            t1 = tio.ScalarImage(tensor=mr_img),
            label = tio.LabelMap(tensor=mr_label)
        )

        # apply image transformation
        # ------------------------------------------------
        if self.aug:
            if self.transform_ct is None:
                ct_img = ct_subject.t1.data
                mr_img = mr_subject.t1.data
                ct_label = ct_subject.label.data
                mr_label = mr_subject.label.data
                # if index % self.ct_size == 0:
                    # save3Dimage_numpy(ct_label.squeeze(), "Images_Print/ct_{}_label".format(index))

            else:
                ct_subject_transformed = self.transform_ct(ct_subject)
                mr_subject_transformed = self.transform_mr(mr_subject)
                print("Aug {} done".format(index))
                ct_img = ct_subject_transformed.t1.data.numpy()
                mr_img = mr_subject_transformed.t1.data.numpy()
                ct_label = ct_subject_transformed.label.data.numpy()
                mr_label = mr_subject_transformed.label.data.numpy()
        
        else:
            orientation_list = [1] if self.test else [0,1,9,10,11]
            if (index % self.mr_size) in orientation_list:
            # if (index % self.mr_size) in [1]:
                # print("Before", ct_img.shape, mr_img.shape)
                mr_img = mr_img[:,112:400,:,112:400]
                mr_label = mr_label[:,112:400,:,112:400]
                mr_img = skTrans.resize(mr_img, (mr_img.shape[0], 256, 256, 256), order=1, preserve_range=True,  anti_aliasing=True)
                mr_label = skTrans.resize(mr_label, (mr_label.shape[0], 256, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
                ct_scale = ct_img.shape[1]/256
                ct_img = skTrans.resize(ct_img, (ct_img.shape[0], int(ct_img.shape[1]/ct_scale), int(ct_img.shape[2]/ct_scale), int(ct_img.shape[3]/ct_scale)), order=1, preserve_range=True,  anti_aliasing=True)
                ct_label = skTrans.resize(ct_label, (ct_label.shape[0], int(ct_label.shape[1]/ct_scale), int(ct_label.shape[2]/ct_scale), int(ct_label.shape[3]/ct_scale)), order=0, preserve_range=True, anti_aliasing=False)
                # print("After", ct_img.shape, mr_img.shape)
            else: 
                # print("Before", ct_img.shape, mr_img.shape)
                # mr_img = mr_img[:,:,mr_img.shape[2]//2-mr_img.shape[1]//2:mr_img.shape[2]//2+mr_img.shape[1]//2,:]
                # mr_label = mr_label[:,:,mr_label.shape[2]//2-mr_label.shape[1]//2:mr_label.shape[2]//2+mr_label.shape[1]//2,:]
                ct_scale = ct_img.shape[1]/256
                mr_scale = mr_img.shape[1]/256
                smaller_than_128 = mr_img.shape[1] < 128
                if ct_scale != 1 or mr_scale!= 1:
                    ct_img = skTrans.resize(ct_img, (ct_img.shape[0], int(ct_img.shape[1]/ct_scale), int(ct_img.shape[2]/ct_scale), int(ct_img.shape[3]/ct_scale)), order=1, preserve_range=True,  anti_aliasing=True)
                    ct_label = skTrans.resize(ct_label, (ct_label.shape[0], int(ct_label.shape[1]/ct_scale), int(ct_label.shape[2]/ct_scale), int(ct_label.shape[3]/ct_scale)), order=0, preserve_range=True, anti_aliasing=False)
                    if smaller_than_128:
                        mr_img = skTrans.resize(mr_img, (mr_img.shape[0], 128, mr_img.shape[2], mr_img.shape[3]), order=1, preserve_range=True,  anti_aliasing=True)
                        mr_label = skTrans.resize(mr_label, (mr_label.shape[0], 128, mr_label.shape[2], mr_label.shape[3]), order=0, preserve_range=True, anti_aliasing=False)
                    # else:
                        # mr_img = skTrans.resize(mr_img, (mr_img.shape[0], int(mr_img.shape[1]/mr_scale), int(mr_img.shape[2]/mr_scale), int(mr_img.shape[3]/mr_scale)), order=1, preserve_range=True,  anti_aliasing=True)
                        # mr_label = skTrans.resize(mr_label, (mr_label.shape[0], int(mr_label.shape[1]/mr_scale), int(mr_label.shape[2]/mr_scale), int(mr_label.shape[3]/mr_scale)), order=0, preserve_range=True, anti_aliasing=False)
                # print("After", ct_img.shape, mr_img.shape)

            # save3Dimage_numpy(ct_label.squeeze(), "Images_Print/ct_{}_label".format(index))
            # save3Dimage_numpy(ct_img.squeeze(), "Images_Print/ct_{}.png".format(index))
            # save3Dimage_numpy(mr_label.squeeze(), "Images_Print/mr_{}_label.png".format(index))
            # save3Dimage_numpy(mr_img.squeeze(), "Images_Print/mr_{}.png".format(index))

        print("index", index, "Completed")

        return {'ct': ct_img, 'mr': mr_img, 'ct_label': ct_label, 'mr_label': mr_label}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.ct_size, self.mr_size)

if __name__ == '__main__':
    counter = 0
    dataroot = sys.argv[1]
    test = sys.argv[2] == "test"
    num_iters = int(sys.argv[3])
    only_resize = sys.argv[2] == "only_resize"
    if test:
        print("Test run")
    if only_resize:
        print("Only resize")
    dataset = DataLoaderDataset(dataroot, no_aug=True, test=test, only_resize=only_resize)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    foler_name = sys.argv[4]
    if test:
        os.makedirs(os.path.join(foler_name, 'testct'), exist_ok = True)
        os.makedirs(os.path.join(foler_name, 'testct_labels'), exist_ok = True)
        os.makedirs(os.path.join(foler_name, 'testmr'), exist_ok = True)
        os.makedirs(os.path.join(foler_name, 'testmr_labels'), exist_ok = True)
    else:
        os.makedirs(os.path.join(foler_name, 'trainct'), exist_ok = True)
        os.makedirs(os.path.join(foler_name, 'trainct_labels'), exist_ok = True)
        os.makedirs(os.path.join(foler_name, 'trainmr'), exist_ok = True)
        os.makedirs(os.path.join(foler_name, 'trainmr_labels'), exist_ok = True)       
    for epoch in tqdm(range(1, num_iters+1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latrain_freq>        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if test:
                np.savez_compressed(os.path.join(foler_name, 'testct', 'ct_test_{}_image.npz'.format(counter)), data["ct"])
                np.savez_compressed(os.path.join(foler_name, 'testct_labels', 'ct_test_{}_label.npz'.format(counter)), data["ct_label"])
                np.savez_compressed(os.path.join(foler_name, 'testmr', 'mr_test_{}_image.npz'.format(counter)), data["mr"])
                np.savez_compressed(os.path.join(foler_name, 'testmr_labels', 'mr_test_{}_label.npz'.format(counter)), data["mr_label"])
            else:
                np.savez_compressed(os.path.join(foler_name, 'trainct', 'ct_train_{}_image.npz'.format(counter)), data["ct"])
                np.savez_compressed(os.path.join(foler_name, 'trainct_labels', 'ct_train_{}_label.npz'.format(counter)), data["ct_label"])
                np.savez_compressed(os.path.join(foler_name, 'trainmr', 'mr_train_{}_image.npz'.format(counter)), data["mr"])
                np.savez_compressed(os.path.join(foler_name, 'trainmr_labels', 'mr_train_{}_label.npz'.format(counter)), data["mr_label"])
            counter += 1
