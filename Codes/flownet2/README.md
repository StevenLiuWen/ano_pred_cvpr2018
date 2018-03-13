## FlowNet2 (TensorFlow)

This repo contains FlowNet2[1] for TensorFlow. It includes FlowNetC, S, CS, CSS, CSS-ft-sd, SD, and 2.

### Installation
```
pip install enum
pip install pypng
pip install matplotlib
pip install image
pip install scipy
pip install numpy
pip install tensorflow
```

Linux:
`sudo apt-get install python-tk`

You must have CUDA installed:
`make all`

### Download weights
To download the weights for all models (4.4GB), run the `download.sh` script in the `checkpoints` directory. All test scripts rely on these checkpoints to work properly.


### Flow Generation (1 image pair)

```
python -m src.flownet2.test --input_a data/samples/0img0.ppm --input_b data/samples/0img1.ppm --out ./
```

Available models:
* `flownet2`
* `flownet_s`
* `flownet_c`
* `flownet_cs`
* `flownet_css` (can edit test.py to use css-ft-sd weights)
* `flownet_sd`

If installation is successful, you should predict the following flow from samples/0img0.ppm:
![FlowNet2 Sample Prediction](/data/samples/0flow-pred-flownet2.png?raw=true)

### Training
If you would like to train any of the networks from scratch (replace `flownet2` with the appropriate model):
```
python -m src.flownet2.train
```
For stacked networks, previous network weights will be loaded and fixed. For example, if training CS, the C weights are loaded and fixed and the S weights are randomly initialized.


### Fine-tuning
TODO

### Benchmarks
Benchmarks are for a forward pass with each model of two 512x384 images. All benchmarks were tested with a K80 GPU and Intel Xeon CPU E5-2682 v4 @ 2.30GHz. Code was executed with TensorFlow-1.2.1 and python 2.7.12 on Ubuntu 16.04. Resulting times were averaged over 10 runs. The first run is always slower as it sets up the Tensorflow Session.

| | S | C | CS | CSS | SD | 2
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| First Run | 681.039ms | 898.792ms | 998.584ms | 1063.357ms | 933.806ms | 1882.003ms |
| Subsequent Runs | 38.067ms | 78.789ms | 123.300ms | 161.186ms | 62.061ms | 276.641ms |


### Sources
[1] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, T. Brox
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks,
IEEE Conference in Computer Vision and Pattern Recognition (CVPR), 2017.
