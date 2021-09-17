## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/orhir/Or-Leo_Final_Prj
cd Or-Leo_Final_Prj
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

## Train
- Train a model:
  -- Phase 1:
```bash
python train.py --dataroot dataset_path/ --model cycle_gan --crop_size_z 32 --crop_size 256 --only_seg --max_dataset_size 200 --name phase_1 --train_phase 1 [--four_labels]
```
  -- Phase 2:
```bash
python train.py --dataroot dataset_path/ --model cycle_gan --crop_size_z 32 --crop_size 256 --load_seg --load_name phase_1 --max_dataset_size 200 --name phase_2 --train_phase 2 [--four_labels]
``` 
  -- Phase 3:
```bash
python train.py --dataroot dataset_path/ --model cycle_gan --crop_size_z 32 --crop_size 256 --load_all_networks --load_name phase_2 --max_dataset_size 200 --name phase_2 --lambda_seg_from_syn 0.5 --train_phase 3 [--four_labels]
``` 

- To see more intermediate results, check out `./checkpoints/MODEL_NAME/web/index.html`.
 - To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

## Test
```bash
python test.py --dataroot test_dataset_path/ --model cycle_gan --load_name phase_3 --crop_size 128 --crop_size_z 64 [--four_labels]
```
- The test results will be saved to a html file here: `./results/[load_name]/latest_test/index.html`.
## Acknowledgement
Part of the code is revised from the [PyTorch implementation of CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Note
* The repository is being updated
