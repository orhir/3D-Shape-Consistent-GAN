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


mr_label = tio.LabelMap("/mnt/storage/datasets/ct_mri_3d_dataset/MM-WHS_2017_Dataset/trainmr_lables/mr_train_1010_label.nii.gz").data
ct_label = tio.LabelMap("/mnt/storage/datasets/ct_mri_3d_dataset/MM-WHS_2017_Dataset/trainct_lables/ct_train_1010_label.nii.gz").data

lables_dict = {
        0 : 0,
        205 : 1,
        420 : 2,
        500 : 3,
        550 : 4,
        600 : 5,
        820 : 6,
        850 : 7
        }

labels_translate = [0, 205, 420, 500, 550, 600, 820, 850]
fix_labels = [421]
# for i in range(len(labels_translate)):
        # label_ct[label_ct == labels_translate[i]] = i
        # label_mr[label_mr == labels_translate[i]] = i
mr_label[mr_label == 421] = 420
print(np.unique(mr_label))
print(np.unique(ct_label))
# save3Dimage_numpy(label, label.shape, "/home/dvir/Leo_Or_proj/ct_to_mri_3d/tests/test.png")
