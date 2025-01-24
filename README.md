### README.md

# **Handwritten Digit Recognition Using Perceptron**

This project implements a Perceptron model to recognize handwritten digits (0-9) using synthetic grayscale images. The model is designed to allow easy customization and future enhancements.

---

## **Features**
1. **Synthetic Dataset Generation**: Generates 20x20 grayscale images for digits (0-9), with slight variations for training and testing.
2. **Perceptron Implementation**:
   - Supports forward propagation, backpropagation, and gradient descent.
   - Includes a sigmoid activation function and cross-entropy loss.
3. **Training and Testing**:
   - Train the Perceptron with user-defined hyperparameters.
   - Test the trained model and evaluate its accuracy and performance using a confusion matrix.

---

## **Project Structure**
```
Neural_Networks/
├── Perceptron/
│   ├── train_digits/             # Generated training images
│   ├── test_digits/              # Generated test images
│   ├── data_preparation.py       # Script for dataset generation and loading
│   ├── perceptron_model.py       # Perceptron class implementation
│   ├── train_perceptron.py       # Script to train the Perceptron
│   ├── test_perceptron.py        # Script to test the Perceptron
│   ├── main.py                   # CLI for running tasks
│   ├── perceptron_model.npz      # Saved trained model (generated after training)
```

---

## **How to Run**

### **1. Generate Dataset**
Run the following to generate training and test images:
```bash
python data_preparation.py
```

### **2. Train the Perceptron**
Train the model with user-defined hyperparameters:
```bash
python train_perceptron.py
```

### **3. Test the Perceptron**
Evaluate the model's accuracy and performance:
```bash
python test_perceptron.py
```

### **4. Use CLI**
Run the `main.py` file to access a menu for generating datasets, training, and testing:
```bash
python main.py
```

---

## **Dependencies**
Install required packages before running the scripts:
```bash
pip install numpy pillow scikit-learn
```

---

## **Note**
- Ensure the `arial.ttf` font file is available for dataset generation.
- The project is modular and allows for adding new activation functions, loss functions, or datasets.

---

## **License**
This project is licensed under the MIT License. Feel free to modify and use it.
