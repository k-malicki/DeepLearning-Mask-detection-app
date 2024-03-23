# Convolutional Neural Network for Face Mask Detection

## Description

In this project, I created a CNN network to detect facial masks in real-time. This model can be useful in security control systems or public health monitoring.

<p align="center">Examples</p>

![mask_detection](https://github.com/k-malicki/DeepLearning-Mask-detection-app/assets/141445691/bf362278-1933-43aa-9774-09985b0f7674)

## Requirements
- Python
- Numpy
- OpenCV
- TensorFlow
- PyQt

To achive the same results please check requirements.txt

The datasets come from this repo: https://github.com/cabani/MaskedFace-Net



## Labels

The model operates on the following labels:

- Mask (Mask): Means that there is a protective mask on the face.
- No_Mask (No mask): Means that there is no mask on the face.
- Covered_Mouth_Chin (Covered mouth and chin): Means that the mouth and chin are covered, but the nose is exposed.
- Covered_Nose_Mouth (Covered nose and mouth): Means that the nose and mouth are covered, but the chin is exposed.


## Usage
- Access Control: Can be used in access control systems to verify that people entering certain areas have their masks on correctly.

## Credits

This project was based on Udemy course:
https://www.udemy.com/course/computer-vision-face-mask-detection-with-deep-learning
