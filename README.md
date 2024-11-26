# Retinal Vessel Segmentation using ViT based Architectures

Russel Abreo, Atharv Shingewar, Anand Patel, Saikiran Kasturi 

<br>

## Basic Info

### Dataset

Retina Blood Vessel <https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel>

<br><br>

### ViT based networks used :

**TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation** \<[arxiv link](https://arxiv.org/abs/2102.04306)\>
<br>
**UNETR: Transformers for 3D Medical Image Segmentation** \<[arxiv link](https://arxiv.org/abs/2103.10504)\>


<br><br>

## Data Preprocessing

- Download Retina Blood Vessel data from <https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel> 
- Save data like below.  

```
# ROOT Directory

retinal-blood-vessels/
├── imgs/
│   ├── train/
│   ├── test/
│   └── val/
└── labels/
    ├── train/
    ├── test/
    └── val/


- Split Data to train, val, test with [data_prep.ipynb](./data_prep.ipynb)

<br>

## Train

```bash
$ python train.py
```

<br>

## Test

```bash
$ python eval.py
```

Then, you can see result images from `./test_results`  

<br><br>

## Reference

- [hanyoseob - youtube-cnn-002-pytorch-unet](https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet)

## Credits

This project builds upon the work of others:

- **Research Paper**:  
  [*TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*](https://arxiv.org/abs/2102.04306) 
```
@misc{chen2021transunettransformersmakestrong,
      title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation}, 
      author={Jieneng Chen and Yongyi Lu and Qihang Yu and Xiangde Luo and Ehsan Adeli and Yan Wang and Le Lu and Alan L. Yuille and Yuyin Zhou},
      year={2021},
      eprint={2102.04306},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2102.04306}, 
}
```

- **Research Paper**:  
  [*UNETR: Transformers for 3D Medical Image Segmentation*](https://arxiv.org/abs/2103.10504) 
```
@misc{hatamizadeh2021unetrtransformers3dmedical,
      title={UNETR: Transformers for 3D Medical Image Segmentation}, 
      author={Ali Hatamizadeh and Yucheng Tang and Vishwesh Nath and Dong Yang and Andriy Myronenko and Bennett Landman and Holger Roth and Daguang Xu},
      year={2021},
      eprint={2103.10504},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2103.10504}, 
}
```

    
- **Code Repository**:  
  Referred TransUNet network from [mkara44](https://github.com/mkara44/transunet_pytorch) and UNET-R from [HXLH50K](https://github.com/HXLH50K/U-Net-Transformer) on GitHub. 
