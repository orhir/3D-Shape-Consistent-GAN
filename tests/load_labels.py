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

for i in range(1, 21):
        if(i<10):
                i_print = "0" + str(i)
        else:
                i_print = str(i)
        mr_label = tio.LabelMap("/mnt/storage/datasets/ct_mri_3d_dataset/MM-WHS_2017_Dataset/trainmr_lables/mr_train_10" + i_print + "_label.nii.gz").data
        ct_label = tio.LabelMap("/mnt/storage/datasets/ct_mri_3d_dataset/MM-WHS_2017_Dataset/trainct_lables/ct_train_10" + i_print + "_label.nii.gz").data

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


        ct_box = [slice(np.min(indexes), \
                np.max(indexes) + 1) for indexes in np.where(ct_label>0)]
        for i in range(1, 3):
                if (ct_box[i].stop - ct_box[i].start + 1) < 128:
                        ct_box[i] = slice(ct_box[i].start, ct_box[i].start + 128)
        if (ct_box[3].stop - ct_box[3].start + 1) < 64:
                ct_box[3] = slice(ct_box[3].start + 1, ct_box[i].start + 64)

        mr_box = [slice(np.min(indexes), \
                np.max(indexes) + 1) for indexes in np.where(mr_label>0)]
        for i in range(1, 3):
                if (mr_box[i].stop - mr_box[i].start + 1) < 128:
                        mr_box[i] = slice(mr_box[i].start, mr_box[i].start + 128)
        if (mr_box[3].stop - mr_box[3].start + 1) < 64:
                mr_box[3] = slice(mr_box[3].start + 1, mr_box[i].start + 64)


        mr_label = mr_label[mr_box]
        ct_label = ct_label[ct_box]
        print(mr_label.shape)
        print(ct_label.shape)
        # print(np.unique(mr_label))
        # print(np.unique(ct_label))

        mr_label = mr_label.squeeze()
        ct_label = ct_label.squeeze()
        save3Dimage_numpy(mr_label, mr_label.shape, "/home/dginzburg/Or-Leo_Final_Prj/tests/mr_label" + str(i) + ".png")
        save3Dimage_numpy(ct_label, ct_label.shape, "/home/dginzburg/Or-Leo_Final_Prj/tests/ct_label" + str(i) + ".png")

