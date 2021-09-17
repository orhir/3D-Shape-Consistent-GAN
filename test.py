"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import util, html
import sklearn.metrics
import numpy as np
import torch
import medpy.metric.binary as mmb


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f==y_pred_f)
    smooth = 1e-7
    return (2. * intersection) / (y_true_f.size + y_pred_f.size + smooth)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.load_name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.load_name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    scores = {}
    labels_translate = [0, 205, 420, 500, 550, 600, 820, 850]
    visuals = {}

    for j, data in enumerate(dataset):
        if j >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        data1 = data.copy()
        data1['mr'] = data1['mr'][:,:,:,:,:data1['mr'].shape[4]//2]
        data1['ct'] = data1['ct'][:,:,:,:,:data1['ct'].shape[4]//2]
        data1['mr_label'] = data1['mr_label'][:,:,:,:,:data1['mr_label'].shape[4]//2]
        data1['ct_label'] = data1['ct_label'][:,:,:,:,:data1['ct_label'].shape[4]//2]

        data2 = data.copy()
        data2['mr'] = data2['mr'][:,:,:,:,data2['mr'].shape[4]//2:]
        data2['ct'] = data2['ct'][:,:,:,:,data2['ct'].shape[4]//2:]
        data2['mr_label'] = data2['mr_label'][:,:,:,:,data2['mr_label'].shape[4]//2:]
        data2['ct_label'] = data2['ct_label'][:,:,:,:,data2['ct_label'].shape[4]//2:]

        model.set_input(data1)  # unpack data from data loader
        print(data1['mr'].shape, data1['ct'].shape)
        model.test()           # run inference
        visuals1 = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        model.set_input(data2)  # unpack data from data loader
        print(data2['mr'].shape, data2['ct'].shape)
        model.test()           # run inference
        visuals2 = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        # Calculate metrics
        for dir in ["A", "B"]:
            seg = np.concatenate((visuals1["seg_" + dir].cpu().detach().numpy(), visuals2["seg_" + dir].cpu().detach().numpy()), axis=4)
            truth = np.concatenate((visuals1["ground_truth_seg_" + dir].cpu().detach().numpy(), visuals2["ground_truth_seg_" + dir].cpu().detach().numpy()), axis=4)
            real = np.concatenate((visuals1["real_" + dir].cpu().detach().numpy(), visuals2["real_" + dir].cpu().detach().numpy()), axis=4)

            # metric = dice_coef(truth, seg)
            dice_list = []
            assd_list = []
            for c in range(1, 8):
                pred_test_data_tr = seg.flatten().copy()
                pred_test_data_tr[pred_test_data_tr != c] = 0
                pred_test_data_tr[pred_test_data_tr == c] = 1

                pred_gt_data_tr = truth.flatten().copy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0
                pred_gt_data_tr[pred_gt_data_tr == c] = 1

                if ( not pred_gt_data_tr.any()):
                    print(c, "is all zeros - ignore")
                else:
                    metric = sklearn.metrics.f1_score(pred_gt_data_tr.flatten(), pred_test_data_tr.flatten(), average="binary")
                    # dice_score = mmb.dc(pred_test_data_tr, pred_gt_data_tr)
                    print(metric)
                    dice_list.append(metric)
                    # assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
                    
            visuals["real_" + dir] = torch.from_numpy(real)
            visuals["seg_" + dir] = torch.from_numpy(seg)
            visuals["ground_truth_seg_" + dir] = torch.from_numpy(truth)

            print ('Mean:%.1f' % np.mean(dice_list))

            # metric = sklearn.metrics.f1_score(truth.flatten(), seg.flatten(), average='samples', sample_weight = np.array([0,1,1,1,1,1,1,1]))
            if dir in scores:
                scores[dir] = np.concatenate((scores[dir], np.array([np.mean(dice_list)])))
            else:
                scores[dir] = np.array([np.mean(dice_list)])


        
        
        if j % 1 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (j, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML
    # print results
    print("\n\n|{}|".format("-"*100))
    print("|" + "-"*42 + "Detaild scores: " + "-"*42 + "|")
    print("|{}|".format("-"*100))
    print("|{}CT  Segmentation F1 Scores: {}{}|".format(" "*13, str(scores["A"]), " "*14))
    print("|{}MRI Segmentation F1 Scores: {}{}|".format(" "*13, str(scores["B"]), " "*14))
    print("|{}|".format("-"*100))
    print("|" + "-"*43+ "Total scores: " + "-"*43 + "|")
    print("|{}|".format("-"*100))
    print("|{}CT  Segmentation F1 Score: {}{}|".format(" "*30,np.mean(scores["A"]), " "*25))
    print("|{}MRI Segmentation F1 Score: {}{}|".format(" "*30,np.mean(scores["B"]), " "*25))
    print("|{}|".format("-"*100))

