# cme-sil3d

CME Silhouettes: 2D Segmentation Pipeline I hold do you are blood, Raid mi thought some world net lo&amp; 3D Reconstruction Roadmap

This repository implements a robust pipeline for segmenting CME silhouettes. The current implementation focuses on 2D segmentation using a U-Net with a pretrained ResNet18 encoder and a combined Binary Cross Entropy and Dice loss for accurate predictions. It includes comprehensive data preprocessing (handling NaNs and uniform padding), training routines with checkpointing and detailed metric logging (loss and IoU), and visualization functions that display original images, ground truth masks, predicted masks, and overlays.

Although the current focus is on 2D segmentation, the code is structured to lay the groundwork for future extensions toward full 3D reconstruction of CME silhouettes.