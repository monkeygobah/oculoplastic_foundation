# Oculoplastic Segmentation Pipeline using Foundational Models

## Overview
This repository contains an end-to-end pipeline for oculoplastic segmentation leveraging foundational models. The pipeline is designed to function without the need for additional training, making it accessible and efficient for practical applications.

## Key Features
1. **End-to-End Pipeline for Oculoplastic Segmentation**:
   - Complete pipeline from bounding box generation to segmentation and distance prediction without requiring any training.

2. **Bounding Box Generation with MediaPipe**:
   - MediaPipe is utilized to generate bounding boxes which are then submitted to the Segment Anything Model (SAM) for segmentation.

3. **Segmentation Masks and Distance Prediction**:
   - Segmentation masks generated by SAM are used to predict various distances relevant to oculoplastic analysis.

4. **Exploration of Augmentation Techniques**:
   - Various augmentation techniques such as super-resolution, cropping, and more are explored in detail to enhance the quality and robustness of the segmentation results.

5. **Plotting and Display Handling**:
   - Comprehensive handling of plotting and displaying results for easy visualization and interpretation.

## Note
- **Ongoing Experiments**:
  - Please note that experiments are still underway. Much of the code is in a preliminary state and requires cleaning up. Skeleton code and a more refined version will be released in the final iteration.
