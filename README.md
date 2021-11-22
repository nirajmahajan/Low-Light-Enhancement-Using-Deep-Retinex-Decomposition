# Low Light Enhancement

PyTorch Implementation of the paper [Deep Retinex Decomposition for Low-Light Enhancement](https://arxiv.org/abs/1808.04560)

Implemented as a course project for GNR 638 (Machine Learning for Remote Sensing) at IIT Bombay.

## Introduction

Retinex models are an effective tool for low-light image enhancement. It assumes that observed images can be decomposed into the reflectance and illumination. In this paper, the authors collect a LOw-Light dataset (LOL) containing low/normal-light image pairs and propose a deep Retinex-Net learned on this dataset, including a Decom-Net for decomposition and an Enhance-Net for illumination adjustment. 

The paper proposes an end-to-end training pipeline, that works without ground truth labels of the reflectance and the illuminance of images, which greatly adds to the ease and convenience of the training process. The Qualitative results obtained on completely unknown images show that the method not only achieves good Low Light Enhancement, but also learns to decompose the image into reflectance and illuminance effectively, in a weakly supervised manner.

![](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/blob/master/images/architecture.png)

## Setup

The code was trained using python3 with a requirement of the following libraries:

1. pytorch - 1.9.0
2. torchvision  - 0.2.1
3. numpy - 1.21.0
4. matplotlib - 3.3.4
5. OpenCV - 4.4.0
6. PIL - 8.2.0
7. tqdm - 4.61.1

## Dataset

The dataset can be downloaded from the following links:

- LOw Light paired dataset (LOL): [Google Drive](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB)
- Synthetic Image Pairs from Raw Images: [Google Drive](https://drive.google.com/open?id=1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F)
- Testing Images: [Google Drive](https://drive.google.com/open?id=1OvHuzPBZRBMDWV5AKI-TtIxPCYY8EW70)

The directory structure should be as follows:

```bash
dataset/
└── Combined/
        ├── test
        │   ├── high
        │   │   ├── 1.png
        │   │   ├── .....
        │   └── low
        │       ├── 1.png
        │       └── ......
        ├── train
        │   ├── high
        │   │   ├── 2.png
        │   │   ├── ......
        │   └── low
        │       ├── 2.png
        │       └── ......
        └── unknown
            ├── high
            │   ├── 1.png
            │   ├── ......
            └── low
                ├── 1.png
                └── ......

```



## Training Method

In the training process for Decom-Net, there is no ground truth of decomposed reflectance and illumination. The network is learned with only key constraints like

1. The original image should be reconstructed using the Reflectance (R) and Illuminance (I)
2. The Reflectance (R) should be shared by paired low/normal-light images
3. The Illumination and the reconstruction should be smooth (constraining the gradient)

Based on the decomposition, subsequent lightness enhancement is conducted on illumination by an enhancement network called Enhance-Net. For joint denoising there is a denoising operation on reflectance.

## Usage of Code

The code can be run using the following command:

```bash
$ cd src/
$ python3 main.py --cuda <gpu id>					# Start Training
# The gpu_id must be set -1 in case training on CPU
$ python3 main.py --cuda <gpu id> --resume 				# Resume Training
$ python3 main.py --cuda <gpu id> --resume --eval 			# Evaluate the model
```

The pre-trained weights are already loaded in the repository.

## Tuning of hyper parameters

We have further tuned the hyper parameters for the training by running over 30 experiments. More details about the experiments can be found [here](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/tree/master/experiments).

## Results and Observations

In this section, we have attached our own results after training the model from scratch. The trained model gave good results in almost all cases. But we also noticed that in very few cases the Illuminance misbehaved, ie, gave extremely high values which generated abnormally bright and cloudy images.

All our generated results can be found [here](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/tree/master/results). We tested the performance on two separate sets of images:

1. Known Images: Test images that are from similar locations as the training data. (Of course they are not used for training)
2. Unknown Images: Completely foreign images

We have attached a few results here:

![](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/blob/master/results/unknown/28.png)

![](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/blob/master/results/unknown/35.png)

![](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/blob/master/results/unknown/45.png)

![](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/blob/master/results/unknown/48.png)

![](https://github.com/nirajmahajan/Low-Light-Enhancement-Using-Deep-Retinex-Decomposition/blob/master/results/unknown/49.png)

## Team

The team for this project

1. [Niraj Mahajan](https://www.cse.iitb.ac.in/~nirajm)
2. [Abhinav Kumar](https://cse.iitb.ac.in/~abhinavkumar)
3. [Nimay Gupta](https://www.cse.iitb.ac.in/~nimay)
4. [Raaghav Raaj Kuchiyaa](https://www.cse.iitb.ac.in/~raaghav)



## References

1. [Chen Wei*](https://weichen582.github.io/), [Wenjing Wang*](https://daooshee.github.io/website/), [Wenhan Yang](https://flyywh.github.io/), [Jiaying Liu](http://www.icst.pku.edu.cn/struct/people/liujiaying.html), Deep Retinex Decomposition for Low-Light Enhancement. BMVC'18 (Oral Presentation)
