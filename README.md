# ANN-INFO7375

In this lab, we've built a **multilayer neural network** designed to classify students as "Pass" or "Fail" based on their scores across **five subjects**. The dataset contains scores of **100 students**, and their average scores decide their pass or fail status (50 or above means Pass; below 50 means Fail).

The neural network structure we used is deeper and consists of multiple layers as specified by the assignment:
- Layer 1: **10 neurons** with ReLU activation
- Layers 2 and 3: **8 neurons** each, both with ReLU activation
- Layer 4: **4 neurons** with ReLU activation
- Output layer: **1 neuron** with Sigmoid activation (for binary classification)

We trained the model with the dataset and visualized its predictive performance using a **confusion matrix**, clearly displaying how accurately the model predicts "Pass" and "Fail" cases.

Finally, the lab includes an **interactive part**, allowing you to input any student's number and subject number to check their specific score, average, and result directly.

This lab work demonstrates your understanding of building, training, and evaluating deep neural networks for a simple and practical binary classification problem.
