# ImageNet-1K-training-script
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


## Setting Up the ILSVRC Dataset for Use with PyTorch

The `torchvision.datasets` library doesn't include the ILSVRC dataset, so you'll need to manually download and set it up for use with the `timm` library. Here's a step-by-step guide to get you started.

### Step 1: Prepare Directory Structure

First up, let's create a dedicated folder for our ImageNet classification project and organize it into training and validation subfolders:

```bash
mkdir ImageNet_classification
cd ImageNet_classification
mkdir train
mkdir val
```

### Step 2: Download the Dataset

Next, grabing the training and validation datasets from the official ImageNet repository:

```bash
# Download the training set
cd train
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# Go back to the base directory
cd ..

# Download the validation set
cd val
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Return to the base directory again
cd ..
```

### Step 3: Extract and Set Up the Data

Finally, you need to extract the downloaded files and set them up:

```bash
# Extract the training data
cd train
tar -xvf ILSVRC2012_img_train.tar

# Go back to the base directory
cd ..

# Extract the validation data and make its format similar to training data
cd val
tar -xvf ILSVRC2012_img_val.tar
./valprep.sh
cd ..

# All done! Your data is ready.
```

## Training models on ImageNet-1K by timm library

### Step 1: Installation of timm (Refer to https://huggingface.co/docs/timm/installation)

Start a virtual environment inside your directory:

```bash
# new python environment for pytorch
conda create -n ImageNet_training

# activate the new environment
conda activate ImageNet_training

# install basic packages for scientific computing
conda install -y numpy matplotlib scipy scikit-learn jupyter ipython pandas

# install timm
pip install timm

```
Alternatively, you can install timm from GitHub directly to get the latest, bleeding-edge version:
```bash
pip install git+https://github.com/rwightman/pytorch-image-models.git
```
Run the following command to check if timm has been properly installed:
```bash
python -c "from timm import list_models; print(list_models(pretrained=True)[:5])"
```
This command lists the first five pretrained models available in timm (which are sorted alphebetically). You should see the following output:
```bash
['adv_inception_v3', 'bat_resnext26ts', 'beit_base_patch16_224', 'beit_base_patch16_224_in22k', 'beit_base_patch16_384']
```

### Step 2: Start training
