#!/usr/bin/env bash
/home/liuwen/ssd/datasets/ped2/training/frames
/home/liuwen/ssd/datasets/ped2/testing/frames

python train.py  --dataset  ped2    \
                 --train_folder  ../Data/ped2/training/frames     \
                 --test_folder  ../Data/ped2/testing/frames       \
                 --gpu  0       \
                 --iters    80000


python inference.py  --dataset  ped2    \
                    --test_folder  /home/liuwen/ssd/datasets/ped2/testing/frames      \
                    --gpu  3    \
                    --snapshot_dir    models/pretrains/ped2


python train.py  --dataset  avenue    \
                 --train_folder  ../Data/avenue/training/frames     \
                 --test_folder  ../Data/avenue/testing/frames       \
                 --gpu  2       \
                 --iters    80000

python inference.py  --dataset  avenue    \
                     --test_folder  ../Data/avenue/testing/frames       \
                     --gpu  3


python train.py  --dataset  ped1    \
                 --train_folder  ../Data/ped1/training/frames     \
                 --test_folder  ../Data/ped1/testing/frames       \
                 --gpu  2       \
                 --iters    80000

python inference.py  --dataset  ped1    \
                     --test_folder  ../Data/ped1/testing/frames       \
                     --gpu  3

python train.py  --dataset  ped1    \
                 --train_folder  ../Data/ped1/training/frames     \
                 --test_folder  ../Data/ped1/testing/frames       \
                 --gpu  0       \
                 --iters    80000   \
                 --config   training_hyper_params/hyper_params_lp_0.ini

python inference.py  --dataset  ped1    \
                     --test_folder  ../Data/ped1/testing/frames       \
                     --gpu  1   \
                     --config   training_hyper_params/hyper_params_lp_0.ini


python inference.py  --dataset  ped2    \
                     --test_folder  /home/liuwen/ssd/datasets/ped2/testing/frames       \
                     --gpu  1   \
                     --snapshot_dir     models/pretrains/ped2