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
- Save the data as the given directory structure below . The code to split the data is present in the notebook. 

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

```

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

### Experiments
![Readme1](https://github.com/user-attachments/assets/67dbd3ea-fcd4-46fb-a8d3-0b7a9608634a)
Result table with #Param and #FLOPs

<br><br>

### Results

![image](https://github.com/user-attachments/assets/bbd6b28c-9bac-42c7-956d-2805e66382fb)
Result table with GPU Consumption and AVG Time

<br><br>

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
  Referred TransUNet network from [mkara44](https://github.com/mkara44/transunet_pytorch) and UNET-R from [HXLH50K](https://github.com/HXLH50K/U-Net-Transformer) on GitHub. Referred techniques of augmentation from [devbruce](https://github.com/devbruce/Segmentation-UNet-PyTorch) and segmentation metrics from [loli](https://github.com/loli/medpy/blob/master/medpy/metric/binary.py)
