import matplotlib.pyplot as plt
import numpy as np
import torchio as tio





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


ct_path = "/mnt/storage/datasets/ct_mri_3d_dataset/original_with_npz/trainct/ct_train_1_image.npz"
ct_img = np.load(ct_path)['arr_0']

mr_path = "/mnt/storage/datasets/ct_mri_3d_dataset/original_with_npz/trainmr/mr_train_1_image.npz"
mr_img = np.load(mr_path)['arr_0']

mr_img = mr_img.squeeze()
ct_img = ct_img.squeeze()
save3Dimage_numpy(mr_img, mr_img.shape, "/home/dginzburg/Or-Leo_Final_Prj/tests/mr_img.png")
save3Dimage_numpy(ct_img, ct_img.shape, "/home/dginzburg/Or-Leo_Final_Prj/tests/ct_img.png")

