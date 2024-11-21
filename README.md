# Sequence Prediction with CNNs: A Novel Machine Learning Approach to Morphology
Authors: Deepankur Jain, Somanshu Rath, Param Gandhi, Saksham
## Abstract

Morphological operations like dilation and erosion are foundational in binary image processing, yet their sequential application to transform one binary matrix into another remains an underexplored domain. This study investigates whether a convolutional neural network (CNN) can predict the sequence of transformations required to iteratively transform a binary input matrix \( A \) into a target output matrix \( B \). The transformations are defined over a hypothesis class comprising 16 distinct operations, derived from applying dilation and erosion with eight unique structuring elements. 

To evaluate this, we developed a synthetic dataset of binary matrix transformations, applying sequences of operations using predefined structuring elements, including cross, plus, rhombus, square, and directional line kernels. A CNN model was trained to classify the next transformation in a sequence, leveraging convolutional layers, batch normalization, and global average pooling for efficient spatial feature extraction. The model's performance was assessed using two metrics: next-in-sequence prediction accuracy and the ability to reconstruct full transformation sequences.

Experimental results demonstrate the model's robustness in generalizing across varying matrix sizes and sequence lengths. Notably, while the next-operation prediction accuracy was limited, the model achieved a remarkable 54.7% success rate in generating complete transformation sequences via alternative paths, often discovering shorter paths to achieve the target matrix. These findings reveal the potential of neural networks to approximate and optimize complex binary matrix transformations, opening avenues for further research into sequence learning and abstraction in image processing tasks.

## Problem Statement

The objective of this project is to investigate the capacity of sequential morphological operations, such as dilation and erosion, to iteratively transform a binary input matrix \( A \) into a target output matrix \( B \). These operations are applied using a set of predefined structuring elements, creating a hypothesis class of 16 distinct transformations. The transformation process can be mathematically represented as:


B = f_n ⚬ f_{n-1} ⚬ ... ⚬ f_1(A), f_i ∈ S


where \( S \) is the set of transformations derived from applying dilation and erosion with eight unique structuring elements.

The project aims to evaluate the hypothesis that a neural network model can predict the sequence of transformations required to iteratively transform \( A \) into \( B \). Specifically, the model is tasked with:

1. Predicting the next operation in the sequence at each step.
2. Reconstructing the full transformation sequence when \( A \) and \( B \) are given.

Performance will be assessed through:

- **Next-in-sequence prediction accuracy**: The model's ability to correctly predict the next operation in a sequence.
- **Full-sequence prediction accuracy**: The model's ability to accurately reconstruct the entire sequence of transformations leading to \( B \).

The study investigates the influence of factors such as the choice of structuring elements, sequence length, and the complexity of the operations on model performance. Additionally, it evaluates the model's ability to generalize to unseen transformation sequences and its robustness when handling longer sequences. The ultimate goal is to determine whether the neural network can effectively learn and predict the transformation process across a diverse set of binary matrices and structuring elements.

## Methodology

The objective of this project is to investigate the impact of image processing operations, particularly the morphological operations "dilation" and "erosion", on binary image sequences. The study focuses on generating a dataset of binary image transformations using different structuring elements (SE) and evaluating the ability of a Convolutional Neural Network (CNN) to predict the next transformation to be applied in sequence.

This study employs an experimental approach centered on sequential image processing. The primary morphological operations tested are dilation and erosion, applied using a predefined set of eight unique structuring elements (SE). The transformations are applied sequentially to a binary matrix \( A \), producing an output matrix \( B \) after multiple steps. Each transformation in the sequence is drawn from a hypothesis class of 16 possible operations (8 dilations and 8 erosions).

A CNN model is trained to predict the next transformation in sequence (morphological operation and structuring element) applied to \( A \) to generate \( B \). The architecture leverages convolutional layers to extract spatial features, followed by fully connected layers for classification. The model outputs one of 16 possible labels corresponding to the transformation to be applied next.

### Dataset Generation

Synthetic datasets were created by performing sequential dilations and erosions on randomly generated binary matrices of fixed size \( N \times N \). The dataset is generated as follows:

- Random binary matrices of size \( N \times N \) serve as the initial input \( A \).
- A sequence of morphological operations is applied iteratively to \( A \) using randomly selected structuring elements, resulting in a final output matrix \( B \).
- Each sample in the dataset consists of an input-output pair \((A, B)\) and the label corresponding to the last transformation applied in the sequence.

Since a generating function was used for the datasets, both training and testing data were generated separately using this function.

The structuring elements used in the dataset are as follows:

- **SE1 (Checkerboard Pattern)**: A kernel with alternating squares, creating a checkerboard effect.
- **SE2 (Cross)**: A cross-shaped kernel, focusing on the vertical and horizontal regions intersecting at the center.
- **SE3 (Inverted Checkerboard)**: A reversed checkerboard pattern compared to SE1, with opposite color alternations.
- **SE4 (Hollow Square)**: A square-shaped kernel with a hollow center, affecting only the border regions.
- **SE5 (Vertical Right Half)**: A kernel that emphasizes the right half of the matrix, split vertically.
- **SE6 (Vertical Left Half)**: A kernel that emphasizes the left half of the matrix, split vertically.
- **SE7 (Horizontal Top Half)**: A kernel that emphasizes the top half of the matrix, split horizontally.
- **SE8 (Horizontal Bottom Half)**: A kernel that emphasizes the bottom half of the matrix, split horizontally.

### Model Architecture

The proposed model is a Convolutional Neural Network (CNN) designed specifically to classify sequential morphological operations on binary matrices. It features a robust architecture that effectively balances spatial feature extraction, computational efficiency, and model generalization.

Key components of the architecture include:

1. **Feature Extraction**: The convolutional layers progressively extract spatial features, crucial for understanding the structural changes induced by morphological operations. Max pooling is employed after the first set of convolutions to downsample feature maps, reducing spatial dimensions and computational complexity while retaining the most significant features.
   
2. **Global Average Pooling**: Following the final convolutional layer, global average pooling condenses the spatial information into a single value per feature map. This operation enhances computational efficiency, reduces the risk of overfitting, and ensures the model focuses on global transformation patterns rather than localized noise.
   
3. **Fully Connected Layers**: The condensed features are passed through two dense layers with 64 and 32 neurons, respectively. These layers refine the feature representation, enabling the model to accurately classify the next operation in the sequence. The final dense layer outputs probabilities across 16 classes (representing the possible morphological operations), using a softmax activation function.

The model is trained using the Adam optimizer with a learning rate of 0.001, and the loss function is sparse categorical cross-entropy. A batch size of 32 and a validation split of 20% were used during training to ensure efficient learning and evaluation.

## Experimental Results and Validation

The model was trained on datasets consisting of fixed-size binary matrices for one set of training data, with each input paired with a corresponding label representing one of 16 possible operations. For training, a total of 10,000 input-output pairs were generated for each sequence length from 1 to the maximum sequence length, denoted as *MaxSeqLen*. This resulted in training sets containing \(10,000 \times \textit{MaxSeqLen}\) samples. Training parameters included 50 epochs of the CNN, a batch size of 32, and a 20% validation split. Testing used 1000 samples.

### Testing Conditions

1. **Fixed MaxSeqLen, Varying Matrix Sizes**: The model was tested on datasets with fixed MaxSeqLen for varying matrix sizes to observe prediction accuracy trends for different input sizes.
   
2. **Varying MaxSeqLen, Fixed Matrix Sizes**: Testing was also performed with variable MaxSeqLen, keeping matrix size fixed, to assess the effects of capping the number of operations on the model and its results.

### Full Sequence Prediction

The model was evaluated by predicting full transformation sequences from input \( A \) to output \( B \). Despite low correlation between the actual "next-in-sequence" operator and the prediction, up to 54.7% of inputs were able to reach the outputs via a different path. The model often found shorter paths to reach the target matrix.

## Conclusion and Future Work

This project explores the use of deep learning for classifying image transformations caused by morphological operations. Although the trained CNN model was able to identify some of the operators, the accuracy was around 25% for most cases, indicating the scope of improvement and experimentation with the model architecture and parameter tuning.

**Key Findings**:
- Accuracy of predictions reached a bottleneck for most datasets and sizes.
- The model found shorter paths to reach the output compared to the actual transformation sequence.

**Future Work**:
- **Dataset Expansion**: Increasing the dataset size and including more complex images or operations could help fine-tune the model.
- **Model Improvements**: Exploring other network architectures or optimization techniques may enhance accuracy.
- **Generalization to Real-world Data**: Applying the model to real-world datasets, such as those in medical or industrial imaging, could help assess its practical applicabilitions.
