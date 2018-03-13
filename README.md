# Future Frame Prediction for Anomaly Detection -- A New Baseline
This repo is the official open source of [Future Frame Prediction for Anomaly Detection -- A New Baseline, CVPR 2018](https://arxiv.org/pdf/1712.09867.pdf) by Wen Liu, Weixinluo, Dongze Lian and Shenghua Gao. 
A **demo** is shown in *https://www.youtube.com/watch?v=M--wv-Y_h0A*. It is implemented in tensorflow. Please follow the instructions to run the code.

#### 1. Installation (Anaconda with python3.6 installation is recommended)
* Install 3rd-package dependencies of python (listed in requirements.txt)
```
numpy==1.14.1
scipy==1.0.0
matplotlib==2.1.2
tensorflow_gpu==1.4.1
tensorflow==1.4.1
Pillow==5.0.0
pypng==0.0.18
scikit_learn==0.19.1
opencv-python==3.2.0.6
```

```shell
pip install -r requirements.txt
```
* Other libraries
```code
CUDA 8.0
Cudnn 6.0
```
#### 2. Download datasets
cd into Data folder of project and run the shell scripts (**ped1.sh, ped2.sh, avenue.sh, shanghaitech.sh**) under the Data folder.
```shell
cd Data
./ped2.sh
./ped1.sh
./avenue.sh
./shanghaitech.sh
```

#### 3. Testing on saved models
* Download the trained models
```shell
cd models
./download_pretrains.sh
```
* Running the sript (as ped2 and avenue datasets for examples) and cd into **Codes** folder at first.
```shell
python inference.py  --dataset  ped2    \
                    --test_folder  ../Data/ped2/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    models/pretrains/ped2
```

```shell
python inference.py  --dataset  avenue    \
                    --test_folder  ../Data/avenue/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    models/pretrains/avenue
```


#### 4. Training from scratch (here we use ped2 and avenue datasets for examples)
* Set hyper-parameters
The default hyper-parameters, such as $\lambda_{init}$, $\lambda_{gd}$, $\lambda_{op}$, $\lambda_{adv}$ and the learning rate of G, as well as D, are all initialized in **training_hyper_params/hyper_params.ini**. 
* Running script (as ped2 or avenue for instances) and cd into **Codes** folder at first.
```shell
python train.py  --dataset  ped2    \
                 --train_folder  ../Data/ped2/training/frames     \
                 --test_folder  ../Data/ped2/testing/frames       \
                 --gpu  0       \
                 --iters    80000
```
* Model selection while training
In order to do model selection, a popular way is to testing the saved models after a number of iterations or epochs (Since there are no validation set provided on above all datasets, and in order to compare the performance with other methods, we just choose the best model on testing set). Here, we can use another GPU to listen the **snapshot_dir** folder. When a new model.cpkt.xxx has arrived, then load the model and test. Finnaly, we choose the best model. Following is the script.
```shell
python inference.py  --dataset  ped2    \
                     --test_folder  ../Data/ped2/testing/frames       \
                     --gpu  1
```
Run **python train.py -h** to know more about the flag options or see the detials in **constant.py**.
```shell
Options to run the network.

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU    the device id of gpu.
  -i ITERS, --iters ITERS
                        set the number of iterations, default is 1
  -b BATCH, --batch BATCH
                        set the batch size, default is 4.
  --num_his NUM_HIS    set the time steps, default is 4.
  -d DATASET, --dataset DATASET
                        the name of dataset.
  --train_folder TRAIN_FOLDER
                        set the training folder path.
  --test_folder TEST_FOLDER
                        set the testing folder path.
  --config CONFIG      the path of training_hyper_params, default is
                        training_hyper_params/hyper_params.ini
  --snapshot_dir SNAPSHOT_DIR
                        if it is folder, then it is the directory to save
                        models, if it is a specific model.ckpt-xxx, then the
                        system will load it for testing.
  --summary_dir SUMMARY_DIR
                        the directory to save summaries.
  --psnr_dir PSNR_DIR  the directory to save psnrs results in testing.
  --evaluate EVALUATE  the evaluation metric, default is compute_auc
```
* (Option) Tensorboard visualization
```shell
tensorboard    --logdir=./summary    --port=10086
```
Open the browser and type **https://ip:10086**. Following is the screen shot of avenue on tensorboard.
![scalars_tensorboard](assets/scalars.JPG)

![images_tensorboard](assets/images.JPG)


#### Notes
The flow loss (temporal loss) module is based on [a TensorFlow implementation of FlowNet2](https://github.com/sampepose/flownet2-tf). Thanks for their nice work.
#### Citation
If you find this useful, please cite our work as follows:
```code
@article{liu2018ano_pred,
Author = {Wen Liu and Weixin Luo and Dongze Lian and Shenghua Gao},
Title = {Future Frame Prediction for Anomaly Detection -- A New Baseline},
Journal = {ArXiv e-prints},
Year = {2017},
Eprint = {arXiv:1712.09867},
}
```
While the open access of CVPR 2018 is available, welcome to cite the CVPR version. 
Please contact with us if you have any questions.
