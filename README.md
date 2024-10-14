
# Face Image Reconstruction Using PCA

This project applies **Principal Component Analysis (PCA)** to a dataset of face images. It performs dimensionality reduction and reconstructs images using varying numbers of eigenvectors. PCA helps reduce the dataset's complexity while preserving essential features for image reconstruction.

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Code](#running-the-code)
- [Data Description](#data-description)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## Introduction

This project explores the use of **PCA** for face image reconstruction. The goal is to reduce the high-dimensional image data using a subset of eigenvectors and reconstruct the images with varying accuracy. PCA helps identify patterns in the dataset and reduces redundancy, enabling efficient processing of large datasets.

---

## Getting Started

Follow these instructions to set up the environment and run the code.

### Prerequisites

Make sure the following libraries are installed:

- **Python 3.x**
- **OpenCV**: For reading images.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualization.
- **Scikit-learn**: For PCA implementation.

Install the dependencies using the following pip commands:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### Running the Code

1. **Clone the repository** (if hosted on GitHub):
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Modify the base path** in the code to point to your dataset directory:
   ```python
   base_path = "C:/Users/shiva/OneDrive/Documents/faceimagesdataset"
   ```

3. **Execute the script**:
   ```bash
   python pca_assignment.py
   ```

---

## Data Description

The dataset consists of **40 folders**, each containing **10 grayscale images** of size **112x92 pixels**. These images are reshaped into 1D vectors to form a matrix, where each row represents a flattened image.

- **Base Path**: The folder containing 40 subdirectories.
- **Image Format**: `.pgm` (grayscale).

After processing, the images are stacked into a matrix with shape `(400, 10304)`, where:
- 400 = Total number of images (40 folders × 10 images per folder).
- 10304 = 112 × 92 pixels per image, reshaped into 1D.

---

## Dependencies

This project relies on the following libraries:

- **OpenCV**: To load and process images in grayscale.
- **NumPy**: For matrix operations and numerical computations.
- **Matplotlib**: To visualize original and reconstructed images.
- **Scikit-learn**: To perform PCA and reduce dimensionality.

---

## How It Works

1. **Load and Preprocess Images**:
   - Iterate over 40 folders, each containing 10 images.
   - Read each image in grayscale and reshape it into a 1D array.
   - Store all reshaped images into a matrix (`image_matrix`).

2. **Calculate the Mean Image**:
   - Compute the mean of all images and subtract it from the data to center the dataset.

3. **Perform PCA**:
   - Extract the top **50 principal components** using `PCA(n_components=50)`.

4. **Reconstruct Images**:
   - Randomly select **3 images** for reconstruction.
   - Reconstruct the images using **5, 10, and 50 eigenvectors** to observe the quality differences.

5. **Visualization**:
   - Display the original and reconstructed images side-by-side using `Matplotlib`.

---

## Results

The following outcomes are generated:

1. **Final Image Matrix Shape**:
   ```bash
   Final matrix shape: (400, 10304)
   ```
   This indicates that the dataset contains 400 images, each reshaped into a vector of 10304 pixels.

2. **Reconstructed Images**:
   - Visualizes original and reconstructed images for comparison.
   - Reconstruction is performed using **5, 10, and 50 eigenvectors**.

---

## Future Improvements

1. **Optimize PCA for Large Datasets**:
   - Use **Incremental PCA** to handle datasets that do not fit into memory.

2. **Automate Dataset Path Handling**:
   - Add a configuration file or command-line argument for setting the dataset path.

3. **Performance Comparison**:
   - Compare PCA-based reconstruction with **autoencoders** or **t-SNE** for better insights.

---

## Conclusion

This project demonstrates how **PCA** can be applied to reduce the dimensionality of face image data and reconstruct images with varying accuracy. It highlights the trade-off between the number of eigenvectors used and the quality of reconstruction, providing insights into the power of dimensionality reduction techniques.
#   F a c e - I m a g e - R e c o n s t r u c t i o n - U s i n g - P C A  
 