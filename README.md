# Heart Disease Prediction Using Neural Networks

This project implements a simple neural network to predict the presence of heart disease based on patient medical data. The dataset is sourced from the **UCI Machine Learning Repository**.Libraries used are Numpy, Pandas, Tensorflow/Keras and Scikit-Learn. For all the runs, accuracy has been over 90%.

The neural network consists of:
1. **Input Layer**: Takes in medical data features.
2. **Hidden Layers**:
   - Two fully connected layers with **ReLU (Rectified Linear Unit)** activation, which helps introduce non-linearity and allows the network to learn complex patterns.
3. **Output Layer**:
   - A single neuron with a **Sigmoid activation function**, which outputs a probability between 0 and 1 for binary classification (heart disease or no heart disease).

- The model is compiled using the Adam optimizer, which adapts the learning rate dynamically to improve convergence.
- **Binary cross-entropy** is used as the loss function since this is a binary classification task.
- The model is trained for **20 epochs** with a batch size of **16**.

**Readme generated with chatgpt**

