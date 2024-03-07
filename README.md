# ImageNet-training-script
ImageNet(ILSVRC-2012) training script by timm library

## Introduction
[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, **ImageNet: A Large-Scale Hierarchical Image Database.** *IEEE Computer Vision and Pattern Recognition (CVPR), 2009.* [pdf](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf) 

[2] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) **ImageNet Large Scale Visual Recognition Challenge.** *International Journal of Computer Vision, 2015.*

[3] R. Wightman. Pytorch image models. https://github.com/rwightman/pytorch-image-models, 2019.


### ImageNet[[1]](1)

[ImageNet](https://image-net.org/index.php) is an image database organized according to the [WordNet](https://wordnet.princeton.edu/) hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research. The data is available for free to researchers for non-commercial use.

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC)[[2]](2) evaluates algorithms for object detection and image classification at large scale. One high level motivation is to allow researchers to compare progress in detection across a wider variety of objects -- taking advantage of the quite expensive labeling effort. Another motivation is to measure the progress of computer vision for large scale image indexing for retrieval and annotation. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images.

### Timm[[3]](3)
Py**T**orch **Im**age **M**odels (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.


## Download ILSVRC dataset in a usable way

Since *torchvision.datasets* doesn't contain ILSVRC dataset, we need to download and configure it by ourselves in an appropriate way to satisfy the requirements of timm library.

Firstly, we create the folder for this project and split the folder for train set and the folder for test set:


```bash
mkdir ImageNet_classification
cd ImageNet_classification
mkdir train
mkdir val
```
Then download training set and validation set on official net:
```bash
cd train
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
cd..
cd val
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
cd ..
```
Decompression training file and configure it in an appropriate way to satisfy the requirements of timm library:
```bash
cd train
tar -xvf ILSVRC2012_img_train.tar
cd ..
cd val
tar -xvf ILSVRC2012_img_val.tar
```


