# WSI-SAM: Multi-resolution Segment Anything Model (SAM) for histopathology whole-slide images

> [WSI-SAM](https://arxiv.org/pdf/2403.09257) has been accepted at MICCAI 2024 COMPAYL Workshop!

## Motivation
<p align="center">
  <img src="demo/fig1.png" alt="fig1">
</p>

## Introduction
The Segment Anything Model (SAM) marks a significant advancement in segmentation models, offering robust zero-shot abilities and dynamic prompting. 
However, existing medical SAMs are not suitable for the multi-scale nature of whole-slide images (WSIs), restricting their
effectiveness. To resolve this drawback, we present WSI-SAM, enhancing SAM with precise object segmentation capabilities for histopathology images using multi-resolution patches, 
while preserving its efficient, prompt-driven design, and zero-shot abilities. 
To fully exploit pretrained knowledge while minimizing training overhead, we keep SAM frozen, introducing only minimal extra parameters and computational overhead.
In particular, we introduce High-Resolution (HR) token, Low-Resolution
(LR) token and dual mask decoder. This decoder integrates the original
SAM mask decoder with a lightweight fusion module that integrates features at multiple scales. Instead of predicting a mask independently, we
integrate HR and LR token at intermediate layer to jointly learn features
of the same object across multiple resolutions. Experiments show that
our WSI-SAM outperforms state-of-the-art SAM and its variants. In particular, our model outperforms SAM by 4.1 and 2.5 percent points on a
ductal carcinoma in situ (DCIS) segmentation tasks and breast cancer
metastasis segmentation task (CAMELYON16 dataset).
<p align="center">
  <img src="demo/fig2.png" alt="fig2">
</p>

## Results
### Box
<p align="center">
  <img src="demo/table1.png" alt="table1">
</p>

### Point
<p align="center">
  <img src="demo/fig4.png" alt="fig4">
</p>

## Installation via pip
