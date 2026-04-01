# Brain Tumor Classification 

## Overview

This project focuses on building a deep learning-based system to classify brain tumors using MRI images. Along with prediction, the system also provides visual explanations using Explainable AI techniques, making it easier to understand how the model makes decisions.

The model classifies MRI scans into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The main idea behind this project is not just to achieve good accuracy, but also to make the model interpretable, especially since it is applied in a sensitive domain like healthcare.

---

## Purpose

Analyzing MRI scans manually is time-consuming and depends heavily on expert knowledge. This project aims to:

- Automate tumor classification using deep learning
- Reduce dependency on manual analysis
- Provide visual explanations for predictions
- Improve trust in AI-based medical systems

---

## Methodology

The overall workflow of the project is as follows:

1. **Data Preprocessing**
   - Images are resized to 224 × 224
   - Pixel values are normalized
   - Data augmentation is applied to improve generalization

2. **Model Development**
   - A CNN model is built and trained from scratch
   - MobileNetV2 is used with transfer learning
   - ResNet50 is tested as an additional deep model

3. **Training Setup**
   - Loss function: Categorical Cross-Entropy
   - Optimizer: Adam
   - Evaluation metric: Accuracy

4. **Explainability**
   - Grad-CAM is used to generate heatmaps
   - These heatmaps highlight important regions in MRI images

5. **Quantitative Analysis**
   - A Focus Score metric is used to compare model attention quality

---

## Models

### CNN (Baseline)
- Simple convolutional network
- Learns features directly from the dataset
- Performance is limited compared to advanced models

### MobileNetV2
- Pretrained on ImageNet
- Fine-tuned for this task
- Efficient and performs well on both accuracy and interpretability

### ResNet50 (Experimental)
- Deep residual network
- Strong feature extraction
- Higher computational requirements

---

## Results

From the experiments:

- The CNN model shows weak and scattered attention in Grad-CAM outputs
- MobileNetV2 produces more focused and meaningful activations
- ResNet50 performs well but generates broader activation regions

Overall, MobileNetV2 gives the best balance between accuracy and interpretability.

---

## Explainability

Grad-CAM is used to visualize which parts of the MRI image the model focuses on while making predictions.

This is important because:
- It helps verify whether the model is looking at the correct region
- It makes the system more transparent
- It increases trust, especially for medical applications

A Focus Score is also used to quantitatively compare how well each model focuses on important regions.

---

## Applications

- Brain tumor detection
- Medical image classification
- Clinical decision-support systems
- AI-assisted diagnostics

---

## Future Work

There are several ways this project can be extended:

- Deploying the model on hardware (FPGA using Vivado)
