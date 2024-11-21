# Sequence Prediction with CNNs: A Novel Machine Learning Approach to Morphology

This project explores the use of convolutional neural networks (CNNs) to classify and predict sequences of morphological operations (e.g., dilation and erosion) on binary matrices. The study investigates whether CNNs can effectively predict transformation sequences and generalize across varying matrix sizes and complexities.

---

## Abstract

Morphological operations are foundational in binary image processing, yet their sequential application to transform one binary matrix into another is underexplored. This project demonstrates:
- Robustness in generalizing across different matrix sizes and sequence lengths.
- The ability to reconstruct full transformation sequences, often discovering shorter, alternative paths.

By utilizing synthetic datasets and a hypothesis class of 16 transformations derived from eight structuring elements, this study provides insights into the potential of CNNs for sequence prediction in image processing.

---

## Problem Statement

The project aims to evaluate whether CNNs can:
1. Predict the next operation in a sequence of morphological transformations.
2. Reconstruct an entire sequence of transformations to iteratively transform a binary matrix \( A \) into \( B \).

Transformations are defined using dilation and erosion with predefined structuring elements such as cross, plus, rhombus, square, and directional line kernels.

---

## Methodology

### Dataset Generation
- **Synthetic Data**: Binary matrices of size \( N \times N \) are transformed using a sequence of morphological operations.
- **Structuring Elements**:
  - Cross, checkerboard, hollow square, and directional line kernels (e.g., vertical, horizontal).

### Model Architecture
The CNN model comprises:
1. **Convolutional Layers**: Extract spatial features with ReLU activation and batch normalization.
2. **Global Average Pooling**: Condense spatial information for computational efficiency.
3. **Dense Layers**: Refine features and classify the next transformation.

---

## Experimental Results

1. **Next-Operation Prediction**:
   - Achieved ~25% accuracy across datasets.
2. **Full-Sequence Prediction**:
   - 54.7% success rate in reconstructing sequences.
   - Frequently identified shorter paths (reduced by ~40%) for longer sequences.

---

## Key Findings

- Accuracy plateaued for most datasets but improved slightly for larger matrix sizes and sequence lengths.
- The model identified multiple valid transformation paths, demonstrating its potential for sequence optimization.

---

## Future Work

1. **Dataset Expansion**: Include more diverse and complex transformations.
2. **Model Improvements**: Explore alternative architectures and hyperparameter tuning.
3. **Real-World Application**: Apply the model to domains like medical imaging and industrial automation.

---

## How to Run

1. **Setup Environment**:
   - Install required dependencies:
     ```bash
     pip install tensorflow numpy matplotlib
     ```
2. **Change configuration of the setting**:
   - Edit variables in `main.py` accordingly.
3. **Run the Model**:
   - Run the model using:
     ```bash
     python3 main.py
     ```
     This will first train the model, then test it.

---
