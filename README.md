# Prediction of 3D Fluorescent Imagery from Brightfield Imagery

> **Project name:**

- **Performance benchmarking over fluorescent imagery reconstruction using Python**

  - [Background](@Background-and-Motivation)
  - [Data](@Data-Observation)
  - [Autoencoder](@Autoencoder)
  - [Pix2Pix](@Pix2Pix)
  - [Installation](@Installation-Requirements)

## Main Project Objective

>Enabling simpler and more efficient solutions for cellular biology research
</br>
<!-- ![image](https://user-images.githubusercontent.com/73793617/148350107-f491fa7c-d45c-478a-b1af-d2699f88c51f.png) -->
<p align="center">
<img src="https://user-images.githubusercontent.com/73793617/148350107-f491fa7c-d45c-478a-b1af-d2699f88c51f.png" width="90%">
</p>

## Background and Motivation
</br>
<!-- ![image](https://user-images.githubusercontent.com/73793617/148352759-c8a0a072-a691-4412-91d2-50389583b0a6.png | width=50%) -->
<p align="center">
<img src="https://user-images.githubusercontent.com/73793617/148352759-c8a0a072-a691-4412-91d2-50389583b0a6.png" width="50%">
</p>
 
Why is predicting 3D fluorescence directly from brightfield images necessary?
* Research promotion
* Prevent damage to the cells
* Reduce cost

## Data Observation
Data source [Allen institute’s](https://github.com/AllenCellModeling/pytorch_fnet) project
</br>

Scope: 
- Approximately 20 different organelles  
- 120 TIFFS for each organelle
- Each TIFF is a 3D image
- Each TIFF can be divided into patches


### Autoencoder

.
### Pix2Pix

.
### Installation Requirements
</br>

```bash 
conda create -n my_env python=3.7
conda activate my_env 
conda install -c anaconda tensorflow-gpu
pip install opencv-python
pip install aicsimageio
pip install matplotlib
pip install patchify
pip install -U scikit-learn
pip install imageio
conda deactivate
```
