# MTT-Net: Multi-scale Tokens-Aware Transformer Network for Multi-region and Multi-sequence MR-to-CT Synthesis in A Single Model


## Dataset
We applied N4 bias field correction to the data and performed registration between MR and CT images. The paired data is stored in the following format:
```
/Datasets/
    ├──Brain_001
      ├── mr.nii.gz
      ├── ct.nii.gz
      ├── mask.nii.gz
    ├──Brain_002
      ├── mr.nii.gz
      ├── ct.nii.gz
      ├── mask.nii.gz
    .
    .

    ├──Pelvis_001
      ├── mr.nii.gz
      ├── ct.nii.gz
      ├── mask.nii.gz
    ├──Pelvis_002
      ├── mr.nii.gz
      ├── ct.nii.gz
      ├── mask.nii.gz
```

## Train
To run the train.py file, you need to set common parameters such as the data storage path and patch size. If you need to use VGG perceptual loss, you can go to the official website and download the pre-trained model of VGG19: vgg19-dcbb9e9d.pth.

## Pretrained models
- [vgg19-dcbb9e9d.pth](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
- [resnet_118_23dataset.pth](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18_23dataset.pth)