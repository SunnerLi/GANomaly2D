# GANomaly2D 
### The Extended Version of GANomaly with Spatial Clue

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.0-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision_sunner-19.4.15-green.svg)]()

![](https://github.com/SunnerLi/GANomaly2D/blob/master/img/structure.png)

Abstraction
---
Anomaly item detection is a critical issue in computer vision. Even though there are some research to solve this problem toward whole patch, these methods doesn't contain the spatial information. The computation is time-consuming if the methods are deployed into practical scenario and check the abnormality patch by patch. In this repository, we purposed **GANomaly2D** to solve the anomaly item recognition problem while preserving the localization information. While the anomaly item occurs in the frame, the anomaly score map will reflect the region rather than only predicting the frame is abnormal or not.    

Install
---
We use `Torchvision_sunner` to deal with data loading. You should install the package from [here](https://github.com/SunnerLi/Torchvision_sunner).    

Structure
---
The GANomaly2D is the 2D version of GANomaly [1]. Moreover, the structure of encoder and decoder is revised from the generator in CycleGAN [2]. We also use PatchGAN to replace the original architecture of discriminator.    

Dataset
---
We test this method toward the `Sunset-bird-fly dataset`. In this dataset, the sunset scene are captured. However, a bird flies through the sky in some frames. There are two domain in this dataset:
* Normal domain: the sunset frame without bird
* Abnormal domain: the sunset frame with bird flying

You can download the dataset from here:
```
https://drive.google.com/drive/folders/1GZHP6_mfXS-hk5y1Gkh8y4SjwMKu7xu6?usp=sharing
```

Usage
---
* Train:
```
python3 train.py --train dataset/normal/ --n_iter 2000 --record 5 --batch_size 8 --r 2
```
* Demo:
```
python3 demo.py --demo dataset/abnormal/ --batch_size 1 --r 2
```

Result
---
![](https://github.com/SunnerLi/GANomaly2D/blob/master/img/training_result.png)

The above image shows the training result. The left figure is the input normal image, the middle figure is the reconstruct image by `G_E` and `G_D`, and the right figure is the anomaly score map. After the iterations of training, the most area of anomaly score map is reduce to 0. Only the score of some patch are high since the region might hard to keep the latent feature consistent.    

![](https://github.com/SunnerLi/GANomaly2D/blob/master/img/demo.png)

The above image illustrates the demo result. The inputs are the image in abnormal domain. As you can see, through the response of anomaly score map at the bottom region is high, some high response at bird region can be found. The **GANomaly2D** can somehow to capture the abnormal region of the bird and give the high score.    

Reference
---
[1] S. Akcay, A. A. Abarghouei, and T. P. Breckon. Ganomaly: Semi-supervised anomaly detection via adversarial training. CoRR, abs/1805.06725, 2018.    
[2] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint, 2017.    