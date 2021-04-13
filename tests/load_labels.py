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


label = tio.LabelMap("/home/dvir/Leo_Or_proj/ct_mri_3d_dataset/MM-WHS_2017_Dataset/trainmr_lables/mr_train_1004_label.nii.gz").data
print(label.max())
# save3Dimage_numpy(label, label.shape, "/home/dvir/Leo_Or_proj/ct_to_mri_3d/tests/test.png")
