# Fashion-MNIST Image Classification with Keras

This repository contains a Jupyter Notebook demonstrating the implementation, training, and analysis of a Deep Neural Network (DNN) designed to classify clothing items from the **Fashion-MNIST** dataset. The project explores neural network architecture tuning, regularization techniques, and performance visualization metrics.

## 🚀 Project Overview
The workflow includes fetching the 70,000 grayscale clothing images, normalizing pixel scaling, constructing highly regularized multilayer perceptrons (MLP), and validating the output through training accuracy/loss progression plots.

## 🛠️ Key Features & Optimization Techniques
* **Data Preprocessing:** Rescaling pixel values from `[0, 255]` to `[0.0, 1.0]` for faster gradient convergence.
* **Batch Normalization:** Applied after dense layers to stabilize internal covariate shift during backpropagation.
* **Dropout Regularization:** Integrating a `0.5` drop probability layer to reduce network dependency variations and combat overfitting.
* **Loss & Metrics:** Outfitted with `sparse_categorical_crossentropy` to accommodate non-one-hot integer category labels alongside standard accuracy evaluations.

## 💻 Tech Stack & Packages
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## 📈 Model Architecture & Parameters
The final deep sequential architecture incorporates **244,554 total parameters**:
1. **Input Layer:** Flattens 28x28 matrix maps to a single vector layer of 784 nodes.
2. **Hidden Layer 1:** 256 Nodes (`ReLU` activation) + Batch Normalization + Dropout (`0.5`).
3. **Hidden Layer 2:** 128 Nodes (`ReLU` activation) + Batch Normalization + Dropout (`0.5`).
4. **Hidden Layer 3:** 64 Nodes (`ReLU` activation) + Batch Normalization + Dropout (`0.5`).
5. **Output Layer:** 10 Nodes (`Softmax` activation) matching the 10 item categories.

## 📊 Sample Training History Preview
The network achieves high generalization metrics within a standard 100-epoch execution threshold:
* **Train Accuracy:** ~89.6% 
* **Validation Accuracy:** ~89.2%

## 📋 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com
   ```
2. Install the missing dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pandas seaborn scikit-learn
   ```
3. Open the notebook (`.ipynb`) in Jupyter Notebook, JupyterLab, or VS Code and execute all cells.
