# AT-Net
Learning to Restore Images Degraded by Atmospheric Turbulence Using Uncertainty, recognized as **Best paper** at IEEE International Conference on Image Processing, 2021


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
```
python train.py --learning_rate 2e-4 --crop_size [256,256] --train_batch_size 2 --epoch_start 0 --lambda_loss 2e-3 --exp_name ./results --lambda_GP 0.0015
```
