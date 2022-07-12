# AT-Net
Learning to Restore Images Degraded by Atmospheric Turbulence Using Uncertainty, recognized as ***Best paper*** at IEEE International Conference on Image Processing, 2021


[Rajeev Yasarla](https://sites.google.com/view/rajeevyasarla/home),  [Vishal M. Patel](https://engineering.jhu.edu/ece/faculty/vishal-m-patel/)

     @InProceedings{9506614,
     author={Yasarla, Rajeev and Patel, Vishal M.},
     booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
     title={Learning to Restore Images Degraded by Atmospheric Turbulence Using Uncertainty}, 
     year={2021},
     pages={1694-1698},
     doi={10.1109/ICIP42928.2021.9506614}
     }


Atmospheric turbulence can significantly degrade the quality of images acquired by long-range imaging systems by causing spatially and temporally random fluctuations in the index of refraction of the atmosphere. Variations in the refractive index causes the captured images to be geometrically distorted and blurry. Hence, it is important to compensate for the visual degradation in images caused by atmospheric turbulence. In this paper, we propose a deep learning-based approach for restring a single image degraded by atmospheric turbulence. We make use of the epistemic uncertainty based on Monte Carlo dropouts to capture regions in the image where the network is having hard time restoring. The estimated uncertainty maps are then used to guide the network to obtain the restored image. Extensive experiments are conducted on synthetic and real images to show the significance of the proposed work.

## Prerequisites:
1. Linux
2. Python 2 or 3
3. Pytorch version >=1.0
4. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)
```
pip install requirements.txt
```

## To train AT-Net
1. Clean face images from Helen and CelebA are aligned and used as input to train AT-Net.
2. Specify train directory and validation directory in train.py lines (57-58)
```
    train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/FFHD_data/images512x512/'
    val_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/Tubfaces89/300M/tubimages/'
```
Note for training you should mention clean images path, train_data.py will generate pairs of turbulence degraded images (refer lines 78-96 in train_data.py ). turbulence degradation parameters can be modified in config_tdrn.py
3. Run the following command to train At-Net
```
python train.py --learning_rate 2e-4 --crop_size [256,256] --train_batch_size 2 --epoch_start 0 --lambda_loss 2e-3 --exp_name ./results --lambda_GP 0.0015
```

## To test AT-Net
```
python test.py --val_dir ="path_test_images" --checkpoint="path_to_models" --exp_name "./results"
```
pretrained models can downloaded from this link [Dropbox](https://www.dropbox.com/s/7k32so1s0pgykil/drive-download-20220704T212509Z-001.zip?dl=0)
